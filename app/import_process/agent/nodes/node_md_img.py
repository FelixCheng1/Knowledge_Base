import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
from minio.deleteobjects import DeleteObject

# 核心改造：导入LangChain工具类和多模块消息模块
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from langchain_core.output_parsers import StrOutputParser
# LLM客户端工具类(核心复用，替代原生OpenAI调用)
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖
from langchain.messages import HumanMessage
# 项目配置
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
# 项目日志工具 
from app.core.logger import logger, node_log, step_log
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt
# MinIO支持的图片格式集合
IMAGE_EXTENSTIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename:str) -> bool:
    """
    判断文件是MinIO支持的图片格式 (后缀不区分大小写)
    :param filename: 文件名(含后缀)
    :return:支持返回Turn，否则False
    """
    return Path(filename).suffix.lower() in IMAGE_EXTENSTIONS

"""
    主要目标：将md中的图片进行单独处理，图片转成对应的语义文本，方便后面进行语义检索
    流程：图片 ——>  MinIO服务器 ——> (上文)图片(下文) ——> 传递到视觉模型 ——> 生成图片总结
            ——> 替换md原有的图片显示 ![图片总结](地址) ——> 修改state["md_content"/"md_path"] ——> 结束
    技术总结： minio re正则 多模态模型 提示词
    实现步骤: 
          1. 进行任务和日志处理 
          2. 进行核心参数校验 [校验md_path/md_content/返回images的文件夹地址]
          3. 查找md中使用的图片和上下文 [传入md_content和images文件夹,返回进行模型访问准备 [(图片名,图片地址,(上文,下文))]]
          4. 进行图片内容总结和处理[调用多模态模型,总结图片内容,最终返回 图片名/总结]
          5. 上传图片到minio服务器,替换图片的本地地址和描述!返回替换后的md_content内容
          6. 备份新的md内容,改为原名称 _new.md
          7. 进行md_path和md_content内容更新(state)
          8. 返回目标结果即可
"""
@step_log("step_1_get_content")
def step_1_get_content(state:ImportGraphState) -> Tuple[str, Path, Path]:
    """
    提取和校验内容，并且返回图片地址
    :param:state
    :return:返回md内容，md地址，images地址
    """
    # 1.获取md文件地址
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError(f"md_path参数错误，请检查输入参数")
    md_file_obj = Path(md_file_path)
    if not md_file_obj.exists():
        raise FileNotFoundError(f"md_path参数错误，请检查输入参数！{md_file_path}")

    # 2.获取读取md_content
    if not state["md_content"]:
        state["md_content"] = md_file_obj.read_text(encoding="utf-8")
    # 3.拼接图片存储地址
    images_dir_obj = md_file_obj.parent / "images"

    return state["md_content"], md_file_obj, images_dir_obj


@step_log("step_2_scan_images")
def step_2_scan_images(md_content:str, images_dir_obj:Path) -> list[Tuple[str,Path,Tuple[str,str]]]:
    """
    扫描md文件中的图片，匹配本地图片文件，并截取图片上下文（前后100字符）
    
    Args：
        md_content: md原文内容
        images_dir_obj: 图片所在目录（Path对象）
    
    Return:
        列表 -> [(图片文件名，图片完整路径，(上文，下文))]
    """
    # 存储最终处理好的图片信息
    image_targets = []
    # 使用pathlib遍历目录
    for image_file in images_dir_obj.iterdir():
        img_name = image_file.name
        # 过滤非图片格式
        if not is_supported_image(img_name):
            logger.warning(f"跳过非图片文件：{img_name}")
            continue
        # 正则匹配 md 中的图片语法：![](图片路径)
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        items = list(pattern.finditer(md_content))
        # 没有匹配到 -> 跳过
        if not items:
            logger.warning(f"图片 {img_name} 未在 MD 中引用，跳过")
            continue
        # 获取图片在 MD 中的位置
        start, end = items[0].span()
        # 截取上下文（前后100字符）
        pre_text = md_content[max(start - 100, 0): start]
        post_text = md_content[end: min(end, len(md_content))]
        context = (pre_text, post_text)
        # 组装结果
        image_targets.append((img_name, image_file, context))

    return image_targets

@step_log("step_3_image_summary")
def step_3_image_summary(image_targets, stem) -> dict: 
    """
    总结图片，生成图片名.png - 对应的图片描述内容

    Args:
        image_targets:图片信息[(文件名, 文件地址, (上文, 下文))]
        stem:文件名 -> prompt需要
    Return: 
        图片总结
    """
    # 1.定义任务字典
    summaries = {}
    # 2.定义任务队列 -> 模型访问队列 -> 限制访问次数
    requestes_limiter = deque()

    for image_file, image_path, context in image_targets:
        # 访问限速
        apply_api_rate_limit(requestes_limiter,max_requests=15)
        # 获取多模态对象
        vm_model = get_llm_client(model=lm_config.lv_model)
        # 准备提示词
        prompt = load_prompt("image_summary",root_folder = stem,image_content = context)
        # 将图片转为base64字符串
        if isinstance(image_path,str):
            image_path = Path(image_path)
        # 转换成字符串 base64
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        message = HumanMessage(
            content=[{
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            },
            {
                "type": "text", "text": prompt
            }]
        )

        # 
        chain = vm_model | StrOutputParser()
        summary = chain.invoke([message])
        summaries[image_file] = summary

    return summaries

@step_log("step_4_upload_images_replace")
def step_4_upload_images_replace(image_summaries:dict, image_targets:tuple,md_content:str,stem:str) -> str:
    """
    将图片上传到MinIO服务器
    同时替代原md中的图片地址和描述内容
    确保任何位置可以访问图片
    Args:
        image_summaries: 图片和总结
        image_targets: 图片名称 地址 上下文
        md_content: 原md内容
        stem: md文件名
    Return:
        替换后的md_content
    """
    # 1. 获取minio客户端对象
    minio_client = get_minio_client()
    # 2. 先清空原有md在minio中的图片
    # minio / 存储桶 / 文件名 / 图片
    object_list = minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
        recursive=True
    )
    # 转换成minio删除对象
    delete_object_list = [
        DeleteObject(obj.object_name)
        for obj in object_list
    ]
    # 调用方法删除
    errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_object_list
    )
    for error in errors:
        logger.warning(f"图片删除失败：{error}")
    # 3. 上传图片到MinIO服务器
    # 定义一个字典存储每张图片的信息 {图片名：minio url}
    images_urls = {}
    for image_file, image_path, _ in image_targets:
        # 联网操作最好进行错误保护
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_file}",
                file_path=image_path,
                content_type="image/jpeg"
            )
            # 拼接完整路径 协议 + 端点 + 桶名 + 对象名
            images_urls[image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.debug(f"完成图片：{image_file}上传，URL：{images_urls[image_file]}")
        except Exception as e:
            logger.exception(f"上传图片失败：{image_file}，失败原因：{e}")
            logger.debug("继续上传下一张图片")
    # 4. 拼接替换的完整资料
    images_info = {}
    for image_file, summary in image_summaries.items():
        images_info[image_file] = (summary,images_urls[image_file])
    # 5. 进行md_content内容替换
    if images_info:
        for image_file, (summary,url) in images_info.items():
            rep = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + ".*?\)")
            md_content = rep.sub(lambda _: f"![{summary}]({url})", md_content)
    logger.debug(f"完成新旧md内容的替换，最新内容：{md_content[:200]}")
    return md_content
@step_log("step_5_backup_md_file")
def step_5_backup_md_file(md_path_obj:Path, new_md_content:str) -> str:
    """
    完成新的md磁盘备份，并且返回新的地址
    新的命名规则：_new.md
    Args:
        md_path: 原md地址
        new_md_content: 替换后的md文件
    Return:
        返回新地址
    """
    new_md_path_obj = md_path_obj.parent / (md_path_obj.stem + "_new" + md_path_obj.suffix)
    # 重新解析时需要删除前一次new md 文件
    if new_md_path_obj.exists():
        new_md_path_obj.unlink()
    new_md_path_obj.write_text(new_md_content,encoding="utf-8")
    return str(new_md_path_obj)

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    """
    # 1. 进行任务和日志处理 
    add_running_task(task_id=state["task_id"],node_name="node_md_img")
    # 2. 进行核心参数校验 [校验md_path/md_content/返回images的文件夹地址]
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    # 3. 查找md中使用的图片和上下文 [传入md_content和images文件夹,返回进行模型访问准备 [(图片名,图片地址,(上文,下文))]]
    image_targets = step_2_scan_images(md_content,images_dir_obj)
    # 4. 进行图片内容总结和处理[调用多模态模型,总结图片内容,最终返回 图片名/总结]
    image_summaries = step_3_image_summary(image_targets,md_path_obj.stem)
    # 5. 上传图片到minio服务器,替换图片的本地地址和描述!返回替换后的md_content内容
    new_md_content = step_4_upload_images_replace(image_summaries,image_targets,md_content,md_path_obj.stem)
    # 6. 备份新的md内容,改为原名称 _new.md
    new_md_file_path_str = step_5_backup_md_file(md_path_obj,new_md_content)
    # 7. 进行md_path和md_content内容更新(state)
    state["md_content"] = new_md_content
    state["md_path"] = new_md_file_path_str
    # 8. 返回目标结果即可
    add_done_task(task_id=state["task_id"],node_name="node_md_img")
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 —— 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

      # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")