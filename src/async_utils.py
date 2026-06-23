"""
异步处理工具模块。

提供 asyncio 支持，包括：
- 异步任务队列
- 异步执行器
- 异步缓存操作
- 异步 LLM 调用封装
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time


logger = logging.getLogger(__name__)


class AsyncTaskQueue:
    """
    异步任务队列。
    
    支持：
    - 任务优先级
    - 任务超时控制
    - 任务结果收集
    - 并发限制
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        default_timeout: float = 30.0,
    ):
        """
        初始化异步任务队列。
        
        Args:
            max_concurrent: 最大并发任务数
            default_timeout: 默认任务超时时间（秒）
        """
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, Any] = {}
        self._running = False
    
    async def start(self):
        """启动任务队列处理。"""
        if self._running:
            return
        
        self._running = True
        logger.info(f"异步任务队列启动，最大并发：{self.max_concurrent}")
        
        # 启动处理循环
        asyncio.create_task(self._process_loop())
    
    async def stop(self):
        """停止任务队列处理。"""
        self._running = False
        
        # 取消所有运行中的任务
        for task_id, task in self._running_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"取消任务：{task_id}")
        
        # 等待所有任务完成
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        
        logger.info("异步任务队列已停止")
    
    async def submit(
        self,
        task_id: str,
        coro: Coroutine,
        priority: int = 0,
        timeout: Optional[float] = None,
    ) -> str:
        """
        提交异步任务。
        
        Args:
            task_id: 任务 ID
            coro: 协程对象
            priority: 优先级（数字越小优先级越高）
            timeout: 超时时间（秒），None 使用默认值
            
        Returns:
            任务 ID
        """
        if task_id in self._running_tasks:
            logger.warning(f"任务 {task_id} 已在运行中")
            return task_id
        
        timeout = timeout or self.default_timeout
        
        # 包装任务
        wrapped_task = self._wrap_task(task_id, coro, timeout)
        
        # 添加到优先级队列
        await self._queue.put((priority, task_id, wrapped_task))
        
        logger.debug(f"任务已提交：{task_id}，优先级：{priority}")
        return task_id
    
    async def _wrap_task(
        self,
        task_id: str,
        coro: Coroutine,
        timeout: float,
    ) -> Any:
        """
        包装任务，添加超时和错误处理。
        
        Args:
            task_id: 任务 ID
            coro: 协程对象
            timeout: 超时时间
            
        Returns:
            任务结果
        """
        async with self._semaphore:
            try:
                start_time = time.time()
                result = await asyncio.wait_for(coro, timeout=timeout)
                duration = time.time() - start_time
                
                self._results[task_id] = {
                    "status": "success",
                    "result": result,
                    "duration": duration,
                }
                
                logger.debug(f"任务完成：{task_id}，耗时：{duration:.2f}s")
                return result
                
            except asyncio.TimeoutError:
                self._results[task_id] = {
                    "status": "timeout",
                    "error": f"任务超时（{timeout}s）",
                }
                logger.error(f"任务超时：{task_id}")
                raise
                
            except asyncio.CancelledError:
                self._results[task_id] = {
                    "status": "cancelled",
                    "error": "任务被取消",
                }
                logger.info(f"任务被取消：{task_id}")
                raise
                
            except Exception as e:
                self._results[task_id] = {
                    "status": "error",
                    "error": str(e),
                }
                logger.error(f"任务失败：{task_id}，错误：{e}", exc_info=True)
                raise
    
    async def _process_loop(self):
        """任务处理循环。"""
        while self._running:
            try:
                # 从队列获取任务
                priority, task_id, coro = await self._queue.get()
                
                # 创建任务
                task = asyncio.create_task(coro)
                self._running_tasks[task_id] = task
                
                # 任务完成后清理
                task.add_done_callback(
                    partial(self._task_done_callback, task_id)
                )
                
                self._queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"任务处理循环错误：{e}", exc_info=True)
                await asyncio.sleep(1)
    
    def _task_done_callback(self, task_id: str, task: asyncio.Task):
        """
        任务完成回调。
        
        Args:
            task_id: 任务 ID
            task: 完成的任务
        """
        if task_id in self._running_tasks:
            del self._running_tasks[task_id]
    
    async def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务结果。
        
        Args:
            task_id: 任务 ID
            
        Returns:
            任务结果字典，包含 status、result/error、duration
        """
        return self._results.get(task_id)
    
    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        等待任务完成并返回结果。
        
        Args:
            task_id: 任务 ID
            timeout: 等待超时时间
            
        Returns:
            任务结果字典
        """
        task = self._running_tasks.get(task_id)
        if not task:
            return self._results.get(task_id)
        
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"等待任务超时：{task_id}")
        
        return self._results.get(task_id)
    
    def get_queue_size(self) -> int:
        """获取队列中待处理任务数。"""
        return self._queue.qsize()
    
    def get_running_count(self) -> int:
        """获取正在运行的任务数。"""
        return len(self._running_tasks)


class AsyncExecutor:
    """
    异步执行器。
    
    提供同步代码调用异步方法的桥梁。
    """
    
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化异步执行器。
        
        Args:
            loop: 事件循环，None 则使用当前循环
        """
        self._loop = loop
    
    def run_sync(self, coro: Coroutine) -> Any:
        """
        同步执行异步协程。
        
        Args:
            coro: 协程对象
            
        Returns:
            协程结果
        """
        if self._loop and self._loop.is_running():
            # 在已运行的循环中，使用 run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result()
        else:
            # 创建新的事件循环
            return asyncio.run(coro)
    
    async def run_in_executor(
        self,
        func: Callable,
        *args,
        executor: Optional[ThreadPoolExecutor] = None,
        **kwargs,
    ) -> Any:
        """
        在线程池中执行同步函数。
        
        Args:
            func: 同步函数
            *args: 函数参数
            executor: 线程池执行器，None 使用默认执行器
            **kwargs: 函数关键字参数
            
        Returns:
            函数结果
        """
        loop = asyncio.get_event_loop()
        func_with_kwargs = partial(func, **kwargs) if kwargs else func
        return await loop.run_in_executor(executor, func_with_kwargs, *args)


class AsyncCacheWrapper:
    """
    异步缓存包装器。
    
    为同步缓存操作提供异步接口。
    """
    
    def __init__(self, cache_instance):
        """
        初始化异步缓存包装器。
        
        Args:
            cache_instance: 缓存实例（如 RedisCache）
        """
        self._cache = cache_instance
        self._executor = AsyncExecutor()
    
    async def get(self, key: str) -> Optional[Any]:
        """异步获取缓存值。"""
        return await self._executor.run_in_executor(self._cache.get, key)
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None):
        """异步设置缓存值。"""
        await self._executor.run_in_executor(self._cache.set, key, value, expire)
    
    async def delete(self, key: str):
        """异步删除缓存。"""
        await self._executor.run_in_executor(self._cache.delete, key)
    
    async def exists(self, key: str) -> bool:
        """异步检查键是否存在。"""
        return await self._executor.run_in_executor(self._cache.exists, key)


class AsyncLLMWrapper:
    """
    异步 LLM 调用包装器。
    
    将同步的 LangChain LLM 调用转换为异步调用。
    """
    
    def __init__(self, llm_instance):
        """
        初始化异步 LLM 包装器。
        
        Args:
            llm_instance: LangChain LLM 实例
        """
        self._llm = llm_instance
        self._executor = AsyncExecutor()
    
    async def invoke(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """
        异步调用 LLM。
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            LLM 响应
        """
        return await self._executor.run_in_executor(
            self._llm.invoke,
            messages,
            **kwargs,
        )
    
    async def agenerate(self, prompts: List[str], **kwargs) -> Any:
        """
        异步批量生成。
        
        Args:
            prompts: 提示词列表
            **kwargs: 其他参数
            
        Returns:
            生成结果
        """
        # 如果 LLM 支持原生异步方法，优先使用
        if hasattr(self._llm, "agenerate"):
            return await self._llm.agenerate(prompts, **kwargs)
        
        # 否则使用线程池
        return await self._executor.run_in_executor(
            self._llm.generate,
            prompts,
            **kwargs,
        )


# 全局异步任务队列实例
_async_task_queue: Optional[AsyncTaskQueue] = None


def get_async_task_queue() -> AsyncTaskQueue:
    """获取全局异步任务队列实例。"""
    global _async_task_queue
    if _async_task_queue is None:
        _async_task_queue = AsyncTaskQueue()
    return _async_task_queue


def init_async_task_queue(
    max_concurrent: int = 10,
    default_timeout: float = 30.0,
) -> AsyncTaskQueue:
    """初始化全局异步任务队列。"""
    global _async_task_queue
    _async_task_queue = AsyncTaskQueue(
        max_concurrent=max_concurrent,
        default_timeout=default_timeout,
    )
    return _async_task_queue


async def run_async_task(
    task_id: str,
    coro: Coroutine,
    priority: int = 0,
    timeout: Optional[float] = None,
) -> str:
    """
    便捷函数：提交异步任务。
    
    Args:
        task_id: 任务 ID
        coro: 协程对象
        priority: 优先级
        timeout: 超时时间
        
    Returns:
        任务 ID
    """
    queue = get_async_task_queue()
    return await queue.submit(task_id, coro, priority, timeout)


async def wait_async_task(
    task_id: str,
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    便捷函数：等待异步任务完成。
    
    Args:
        task_id: 任务 ID
        timeout: 等待超时时间
        
    Returns:
        任务结果
    """
    queue = get_async_task_queue()
    return await queue.wait_for_task(task_id, timeout)
