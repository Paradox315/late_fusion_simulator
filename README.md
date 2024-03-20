# 后期融合模拟器

后期融合环境的模拟器，用于测试后期融合算法的性能。

# 环境配置

## conda环境

使用 conda 创建虚拟环境并通过 `requirements.txt` 安装依赖

```bash
conda create -n rrwn python=3.8
conda activate rrwn
pip install -r requirements.txt
```

## poetry环境

使用poetry创建虚拟环境并安装依赖

```bash
poetry install
```
# 项目结构

```bash
   .
├── README.md
├── data
├── notebooks
├── requirements.txt
└── src
    ├── main.py
    ├── models
    │   ├── agent.py
    │   ├── frame.py
    │   └── scenario.py
    ├── numpy_rrwn.py
    ├── runner
    │   └── runner.py
    ├── tests
    │   ├── agent_tests.py
    │   ├── algo_tests.py
    │   ├── environment_tests.py
    │   ├── frame_tests.py
    │   └── tools_tests.py
    └── utils
        ├── CBM.py
        ├── lap.py
        └── tools.py 
```

目录说明：
> - `data`：存放实验结果
> - `notebooks`：存放实验结果分析的notebooks
> - `requirements.txt`：项目依赖
> - `src`：源代码
    >

- `main.py`：主程序

> - `models`：模拟器主要的模型
    >

- `agent.py`：agent模型, 包括agent的参数和动作

>     - `frame.py`：帧模型，包括帧的信息和帧的更新
>     - `scenario.py`：场景模型，包括场景的背景信息和目标信息
>   - `numpy_rrwn.py`：numpy实现的RRWN算法
>   - `runner`：模拟器的主要运行逻辑
>   - `tests`：单元测试代码
>   - `utils`：工具方法和算法实现

# 模拟器运行逻辑

模拟器的运行逻辑如下：

1. 初始化模拟器环境
    1. 加载初始帧的场景（包括背景信息、目标信息和agent信息）
    2. 加载模拟的参数（包括模拟的时间、agent的参数等）
2. 配置环境的噪声信息
3. 指定使用的算法
4. 运行模拟器
    1. 每个时间步，agent根据当前帧的信息和算法，选择动作
    2. agent执行动作，模拟器更新帧的信息
    3. 记录帧的信息
5. 保存模拟结果