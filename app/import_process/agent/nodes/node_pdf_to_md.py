import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config

"""
   node_pdf_to_md 
     参数： state [is_pdf_read_enabled = True | pdf_path = xxx.pdf | local_dir = output ] 
     返回： state [md_path = 地址 | md_content = 内容 ]
     1. 日志和任务状态
     2. step_1_validate_paths路径校验
     3. step_2_upload_and_poll minerU的交互
     4. step_3_download_and_extract 下载和解压
     5. 日志和任务状态 return state 
   step_1_validate_paths
     参数：state pdf_path = xxx.pdf | local_dir = output
     返回： pdf_path_obj Path  local_dir_obj Path 
     1. 非空校验 
     2. 文件校验 pdf_path_obj 没有抛异常 local_dir_obj 没有给与默认
     3. 返回完成可用的Path对象即可
   step_2_upload_and_poll
     参数：pdf对应Path  pdf_path_obj
     返回：str zip url地址
     1. 进行申请，获取要上传文件的地址 
     2. 进行文件上传 session | requests.put 
     3. 轮询获取返回结果 zip_url  （确定一个最大等待时间 1页pdf 1s 间隔时间3 错误码 200 -》 500能容忍）
     4. 返回地址即可
   step_3_download_and_extract
     参数：zip_url , out_dir_obj , 原文件名 path.stem
     返回：解压后的.md的str地址 
     1. zip下载 get    output / stem_result.zip
     2. 检查解压的文件夹地址  output / stem 
     3. 检查解压的文件夹进行防重复处理
     4. 进行解压 zipFile  extractall(解压的目标文件夹)
     5. 考虑文件名字 原文件件名 还是 full 还是其他 
     6. 重命名处理
     7. 路径转成字符串 获取绝对路径最终返回即可！
"""
@step_log("step_1_validate_paths")
def step_1_validate_paths(state: ImportGraphState) -> tuple[Path, Path]:
    """
    步骤1：路径校验与初始化
    校验PDF输入文件与输出目录的有效性。遵循【输入严格校验，输出自动修复】的鲁棒性原则
    1.校验PDF路径非空且文件真实存在，不存在则直接抛异常
    2.校验输出目录，为空赋予默认值，不存在则自动创建
    3.统一转换为Path对象处理，保证路径操作的规范性与跨平台兼容性
    :param state: 流程状态字典，包含pdf_path，local_dir
    :return:无返回值，修改状态字典
    """
    # 1.获取路径参数
    pdf_path =state.get("pdf_path","").strip()
    local_dir = state.get("local_dir","").strip()
    # 2.参数非空校验
    if not pdf_path:
        raise ValueError("pdf_path 不能为空，请提供有效的PDF文件路径")
    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = local_dir
        logger.warning(f"未指定输出目录，使用默认路径：{local_dir}")
    # 3.统一转换为Path对象，标准化路径处理
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    # 4.路径有效性校验（差异化处理：输入严格校验，输出自动修复）
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path_obj}，请检查文件路径是否正确")
    if not local_dir_obj.exists():
        logger.warning(f"输出目录不存在，自行创建：{local_dir_obj}")
        local_dir_obj.mkdir(parents=True,exist_ok=True)
    return pdf_path_obj,local_dir_obj


@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj:Path, local_dir_obj:Path) -> str:
    """
    步骤2：上传PDF至MinerU并轮询解析任务状态
    核心流程：配置校验 ——> 获取上传连接 ——> 文件上传(含重试) ——> 任务轮询(直到完成/失败/超时)
    :param:pdf_path_obj ——> 以校验的的PDF Path对象； local_dir_obj ——> 输出目录Path对象
    :return:解析结果ZIP包的下载链接full_zip_url
    异常：ValueError(配置确实),RuntimeError(请求/上传失败),TimeoutError(任务超时)
    """
    # 1.前期配置校验，拦截无效配置
    if not mineru_config.base_url or not mineru_config.api_key:
        raise ValueError("MinerU的base_url或api_key未配置，请检查配置文件！")

    # 2.构造请求头，调用批量接口获取预签名上传地址与批次ID
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mineru_config.api_key}"
    }
    data = {
        "files": [
            {"name":f"{pdf_path_obj.name}"}
        ],
        "model_version":"vlm"
    }
    # 发起获取上传链接请求
    response = requests.post(url=url,json=data,headers=header)
    if response.status_code != 200:
        raise RuntimeError(f"请求MinerU服务失败，状态码：{response.status_code}，响应内容：{response.text}")

    result = response.json()
    if result["code"] != 0:
        raise RuntimeError(f"MinerU接口返回失败，code：{result["code"]}，msg：{result["msg"]}")

    # 获取上传batch—id与urls
    batch_id = result["data"]["batch_id"]
    signed_url = result["data"]["file_urls"][0]

    # 3. 读取PDF文件二进制内容
    file_data = pdf_path_obj.read_bytes()

    # 4. 使用稳定的Session上传文件(关闭系统环境变量，避免OSS预签名URL校验失败)
    with requests.Session() as session:
        session.trust_env = False
        put_res =session.put(url=signed_url,data=file_data,timeout=60)

        if put_res.status_code != 200:
            raise RuntimeError(f"PDF文件上传失败，状态码：{put_res.status_code}，响应：{put_res.text}")

    # 5. 轮询解析任务状态(带超时控制 + 服务端异常自动重试)
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds =600  #最大超时时间10分钟
    poll_interval =3      #轮询间隔3秒
    start_time = time.time()

    logger.debug("开始轮询解析MinerU解析结果")

    while True:
        # 超时判断
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"MinerU解析任务超时（超时{timeout_seconds}秒），请检查服务或文件大小")
        try:
            poll_res = requests.get(url=poll_url,headers=header,timeout=10)
        except Exception as e:
            logger.warning(f"轮询请求异常：{str(e)}，将重试")
            time.sleep(poll_interval)
            continue
        status_code = poll_res.status_code
        # 服务端5xx异常 -> 降级重试
        if status_code != 200:
            if 500 <= status_code <600:
                logger.warning(f"MinerU服务端异常{status_code}，休眠后自动重试")
                time.sleep(poll_interval)
                continue
            else:
                raise RuntimeError(f"轮询请求失败，客户端异常{status_code}，请检查API_KEY与服务端地址")
        # 解析业务响应结果
        poll_data =poll_res.json()
        if poll_data["code"] != 0:
            raise RuntimeError(f"MinerU轮询接口异常，code：{poll_data["code"]}，msg：{poll_data["msg"]}")
        extract_result = poll_data["data"]["extract_result"]
        if not extract_result:
            logger.debug("暂无解析结果，继续轮询.......")
            time.sleep(poll_interval)
            continue
        # 6. 根据任务状态执行逻辑
        task_state = extract_result[0]["state"]
        if task_state == "done":
            full_zip_url = extract_result[0]["full_zip_url"]
            if not full_zip_url:
                raise RuntimeError("MinerU解析完成，但未返回有效的ZIP下载链接")
            logger.info("PDF解析任务完成，准备下载结果包")
            return full_zip_url
        elif task_state == "failed":
            raise RuntimeError("MinerU解析任务执行失败，请检查PDF文件是否损坏或内容异常")
        else:
            logger.debug(f"任务处理中，当前状态：{task_state}，继续轮询...")
            time.sleep(poll_interval)



@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url:str, output_dir_obj:Path, stem:str) -> str:
    """
    步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件(重命名)
    核心流程：下载ZIP -> 清理旧目录并解压 -> 查找MD文件(按优先级) -> 重命名统一为PDF同名
    参数：zip_url：zip包下载链接，output_dir_obj：输出目录Path，stem：PDF无后缀名称
    返回：最终MD文件的字符串格式绝对路径
    异常：RuntimeError(下载失败)，FileNotFoundError(MD文件)
    """
    # 1.下载zip_url对应的资源（关闭系统代理，避免代理中断CDN的SSL连接）
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(zip_url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"ZIP包下载失败，状态码：{response.status_code}，响应：{response.text}")
    # 2.保存zip文件
    zip_save_path = output_dir_obj / f"{stem}_result.zip"
    zip_save_path.write_bytes(response.content)
    # 3.清理旧目录并解压zip包
    extract_target_dir = output_dir_obj / stem
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)
        logger.warning(f"已存在解压后文件包：{extract_target_dir}，现已删除")
    # 重新创建
    extract_target_dir.mkdir(parents=True,exist_ok=True)
    # 利用zipFile解压
    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_target_dir)
    # 4.处理下md文件，统一姓名，并返回md的字符串地址
    md_file_list: list[Path] = list(extract_target_dir.rglob("*.md"))
    if not md_file_list or len(md_file_list) == 0:
        raise FileNotFoundError("未找到对应的PDF文件")
    target_md_file = None
    # 先读同名的
    for md_file in md_file_list:
        if md_file.stem == stem:
            target_md_file = md_file
            break
    # 再读full.md
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break

    if not target_md_file:
        target_md_file = md_file_list[0] 
    # 不是原名字时进行重名
    if target_md_file.stem != stem:
        # target_md_file.with_name(f"{stem}.md") 修改Path对象 (不涉及文件操作)
        # Path.rename(新路径) 将文件 / 目录从原路径**重命名 / 移动**至新路径，实际修改文件系统
        new_md_file = target_md_file.with_name(f"{stem}.md")
        target_md_file.rename(new_md_file)
        # 关键：rename() 不会更新原Path对象，需重新指向新路径
        target_md_file = new_md_file
    # 解析为绝对路径并返回字符串
    final_md_str_path = str(target_md_file.resolve())
    return final_md_str_path

@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    """
    # 1. 日志和任务状态
    add_running_task(state["task_id"],"node_pdf_to_md")
    # 2. step_1_validate_paths路径校验
    pdf_path_obj, local_dir_obj = step_1_validate_paths(state)
    # 3. step_2_upload_and_poll minerU的交互
    zip_url = step_2_upload_and_poll(pdf_path_obj, local_dir_obj)
    # 4. step_3_download_and_extract 下载和解压
    md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
    # 5.处理响应结果
    state["md_path"] = md_path
    with open(md_path, "r", encoding="utf-8") as f:
        state["md_content"] = f.read()
    add_done_task(state["task_id"],"node_pdf_to_md")
    return state

if __name__ == "__main__":
    # 单元测试：验证PDF转MD全流程
    logger.info("============= 开始node_pdf_to_md节点单元测试===============")

    logger.info(f"测试获取根地址{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc","hak180使用说明书.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    #构造测试状态
    test_state = create_default_state(
        task_id = "test_pdf2md_task_001",
        pdf_path = test_pdf_path,
        local_dir = os.path.join(PROJECT_ROOT,"output")
    )

    node_pdf_to_md(test_state)

    logger.info("========== 结束node_pdf_to_md节点单元测试 ================")