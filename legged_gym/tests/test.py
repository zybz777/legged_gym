import multiprocessing


# 定义一个任务函数
def task_function(arg):
    return arg * 2


if __name__ == "__main__":
    # 创建一个具有4个进程的进程池
    pool = multiprocessing.Pool(processes=4)

    # 提交任务到进程池中并执行
    tasks = range(10)  # 假设有10个任务
    result = pool.map(task_function, tasks)

    # 关闭进程池，防止新任务的提交
    pool.close()

    # 主进程等待所有子进程结束
    pool.join()

    print("所有任务执行完毕.")
    print("结果列表:", result)

    # 清空进程池
    pool.terminate()
