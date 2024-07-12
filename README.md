<h1 align="center">WebCanvas: Streamline Your Web Agent Development and Evaluation</h1>


<p align="center">
  <a href="https://github.com/iMeanAI/WebCanvas/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License MIT"></a>
  <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python Version 3.11"></a>
  <a href="https://github.com/iMeanAI/WebCanvas/issues"><img src="https://img.shields.io/github/issues/iMeanAI/WebCanvas" alt="GitHub Issues"></a>
  <a href="https://github.com/iMeanAI/WebCanvas/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  <a href="https://github.com/iMeanAI/WebCanvas/stargazers"><img src="https://img.shields.io/github/stars/iMeanAI/WebCanvas" alt="GitHub Stars"></a>
  <a href="https://github.com/iMeanAI/WebCanvas/network/members"><img src="https://img.shields.io/github/forks/iMeanAI/WebCanvas" alt="GitHub Forks"></a>

</p>

<p align="center">
    <img src="https://img.icons8.com/color/48/000000/internet.png" alt="Platform" width="15" height="15" style="vertical-align: middle;"/> <a href="https://www.imean.ai/web-canvas">Platform</a> • 
    <img src="https://img.icons8.com/?size=100&id=qGwgMt9xZDy5&format=png&color=000000" alt="Paper" width="17" height="17" style="vertical-align: middle;"/> <a href="https://arxiv.org/abs/2406.12373">Paper</a> • 
    <img src="https://img.icons8.com/color/48/000000/database.png" alt="Dataset" width="15" height="15" style="vertical-align: middle;"/> <a href="https://huggingface.co/datasets/iMeanAI/Mind2Web-Live">Dataset</a> • 
    <img src="https://img.icons8.com/color/48/000000/discord-logo.png" alt="Discord" width="15" height="15" style="vertical-align: middle;"/> <a href="https://discord.gg/dhtgvJ52">Discord</a> • 
    <img src="https://img.icons8.com/?size=100&id=13963&format=png&color=000000" alt="Twitter" width="18" height="18" style="vertical-align: middle;"/> <a href="https://x.com/iMeanAI">Twitter</a> • 
    <img src="https://img.icons8.com/?size=100&id=38288&format=png&color=000000" alt="WeChat" width="18" height="18" style="vertical-align: middle;"/> <a href="https://postimg.cc/4nZ9JYdF">WeChat</a>
</p>

Existing benchmarks for web agent tasks are either offline and static, or operate within a fully reproducible environment with limited Internet dynamics. The WebCanvas project aims to pioneer the online evaluation of web agents. Additionally, we offer a suite of toolkits for scaling and maintaining web agent data to support this endeavor. We welcome any constructive feedback on the project and look forward to partnering with you in developing agents for web tasks!

## 🔥 News

- **[2024, June 18]** Our paper will be presented at [agentic markets workshop](https://sites.google.com/view/amw-2024/home?authuser=0) in ICML 2024 and [natural language reasoning and structured explanations workshop](https://nl-reasoning-workshop.github.io/) in ACL 2024. See you in Vienna and Bangkok!
- **[2024, June 18]** Our pre-print [paper](https://arxiv.org/abs/2406.12373) "WebCanvas: Benchmarking Web Agents in Online Environments" is available!
- **[2024, June 6]** We've released WebCanvas, including [Data](https://huggingface.co/datasets/iMeanAI/Mind2Web-Live), [Platform](https://www.imean.ai/web-canvas), [Toolkits](https://webcanvas.gitbook.io/webcanvas-docs), and Web agents(in this repo)!


## 🌟Features

- **Base Agent Framework**: Includes a universal agent framework with four key modules: Planning, Observation, Memory, and Reward, designed to perform complex tasks within real-world online web environments effectively.
- **Dynamic and Real-time Web Environment Interaction**: Utilizes live web environments to provide a realistic assessment and feedback of web agents.
- **Key Nodes Annotation**: Introduces the concept of "key nodes" to offer in-progress feedback and a granular, phase-based assessment system that adapts to frequent changes in real-world web navigation.
- **Enhanced Granularity of Progress Reward**: Allows for a thorough assessment of the reward module within the framework of autonomous web agents, focusing on the pivotal influence of reward signal quality.
- **Easy to Scale with Online Web Environment**: Connected to a comprehensive suite of toolkits with accurate observation capture and rich action space to define demonstration trajectories and intermediate states for real-time, open-ended web tasks, allowing for robust evaluation in dynamic web environments. Check out our [browser plugin and data platform](https://builder.imean.ai/).
- **Mind2Web-Live Dataset**: Presents a refined version of the original Mind2Web[^1] static dataset, containing 542 tasks with 2439 intermediate evaluation states, serving as the foundation general purpose benchmark.

## 🚀 Roadmap

- **Better Modularity and More Flexible Integration**: To help easier integration of WebCanvas evaluation, connect offline agents to online environment.
- **Better Observation**: Faster in computing, more accurate, and combine more modality(text, code, vision, conversation, etc.)
- **Broader Action Space**: Add actions like cache in memory, output final answer, code execution etc. to develop a better interface for web agent, which may differ from human's.
- **Dynamic Evaluation Function**: Provide toolkit for community to define dynamic evaluation functions(for example, model-based evaluation) as supplementary of current static evaluation functions.
- **More Dataset Coverage**: Introduce more datasets in different domains that address key capabilities in online web tasks.
- **Accumulate Knowledge on Agent Experiences**: Develop better algorithm to handle error encountered when inference in live environment, also accumulate knowledge on agent experiences in different websites.
- **Statistics on Agent Cost other than Performance**: Enable calculation of token consumption or GPU consumption of agent framework or agent model to serve as another optimization goal for truly practical web agent.
- **Cloud Version to Mitigate Environment Discrepancy**: We are working on a cloud version for more reliable evaluation.


## 📋 TODO

- [ ] Support more base model calling(Claude, Gemini, Open-source Models from together.ai, etc.). *(in progress)*
- [ ] Add more benchmarking data as showcase: webarena[^2], GAIA[^3], workarena[^4], etc. *(in progress)*
- [ ] Enable token consumption calculation. *(in progress)*
- [ ] Better modularity to ease integration. *(in progress)*
- [ ] Add vision as an extra observation and implement various grounding strategies.
- [ ] Keep updating error handling module.
- [ ] Develop up-to-date visualizations of current live websites agent performance.
- [ ] Add information extraction related actions and relative evaluation metrics.
- [ ] Enable script-based actions and evaluation.

## 🔍 Evaluation on Existing WebCanvas Benchmarks

### Setting Up the Environment

First, ensure your environment is ready by installing the necessary dependencies:

```bash 
conda create -n webcanvas python=3.11
conda activate webcanvas
pip install -r requirements.txt
```

#### Recommended Environment for Mind2Web-Live

From our experiments, the experimental environment plays a crucial role in agent performance. We recommend experimenting on a Windows server using Chrome or Firefox browser engines, preferably on servers located in the United States. Below is the experiment results on Mind2Web-Live test set.

| Planning Model | IP Region    | System  | Browser | Completion Rate | Task Success Rate | Efficiency Score |
|----------------|--------------|---------|---------|-----------------|-------------------|------------------|
| gpt-3.5-turbo-0125        | United States| Windows | Chrome  | 40.2%           | 16.5%             | 3.03             |
| gpt-3.5-turbo-0125        | United States| Windows | Firefox | 42.1%           | 20.2%             | 2.79             |
| gpt-3.5-turbo-0125        | United States| Linux   | Chrome  | 36.5%           | 15.4%             | 3.33             |
| gpt-3.5-turbo-0125        | United Kingdom| Windows | Chrome | 23.6%           | 8.65%             | 7.78             |
| gpt-3.5-turbo-0125        | Singapore    | Windows | Chrome  | 42.3%           | 21.2%             | 2.95             |


### Configuration

Before running the repos, you need to set up the required API keys as using features dependent on external APIs. Our current framework only supports the OpenAI API. We plan to release updates in the future to support additional models.

For setting up OpenAI API keys, add your API key to your environment variables:

MacOS/Linux:

```
export OPENAI_API_KEY='your-api-key-here'
```

Windows:

```text
setx OPENAI_API_KEY "your-api-key-here"
```

Visit [Quickstart tutorial - OpenAI API](https://platform.openai.com/docs/quickstart?context=python) for more details.


### Download Raw Data of a Challenge

Register on the platform [here](https://www.imean.ai/web-canvas).

First, ensure your environment variables are correctly set so that the code can access the necessary credentials and URL.
```
export GRAPHQL_USERNAME=your_username
export GRAPHQL_PASSWORD=your_password
```

To download a file, use the following command:

```bash
python data/dataset_io.py download \
    --challenge-id your_challenge_id \
    --save-path /path/to/save/file
```

- `your_challenge_id`: The ID of the challenge for the download. Obtain this ID on the url link of the challenge for now. For example, the ID of [Mind2Web-Live Test](https://www.imean.ai/web-canvas/challenges/WjVIjPfpa-psiltU3oD2W/leaderboard) is "WjVIjPfpa-psiltU3oD2W".
- `/path/to/save/file`: The path where the downloaded file will be saved.

#### Process the Raw Data
The raw data contain rich information on step level to inspire future research. However, it's not for our evaluation.

To process the raw data, run the follow command:

```
python data/raw_data_processor.py \
    --input-file path/to/input/file \
    --output-file path/to/output/file
```

### Run the Evaluation

You can run the repos with the following command:

```bash
python evaluate.py \
    --global_reward_mode dom_reward \
    --index -1 \
    --single_task_name "Find Dota 2 game and add all DLC to cart in steam."
```

This command runs the script with DOM-based self-reward, processing the default task "Find Dota 2 game and add all DLC to cart in steam" or using the default data index -1. The evaluation mode is controlled by the `task_mode` parameter in `configs/setting.toml`, allowing you to choose between batch mode and single mode(without automatic evaluation). Remember to specify your path to the test file in `configs/setting.toml`.


### Parameter Descriptions

This program supports several command-line arguments to customize its behavior:

- `--global_reward_mode`: Selects the method for getting global rewards.
  - Options: `dom_vision_reward`, `dom_reward`, `vision_reward`, `no_global_reward`
  - Default: `dom_reward`
  - Description: Define how rewards are got based on the interaction mode:
    - `dom_vision_reward`: Rewards are calculated using both DOM and vision data.
    - `dom_reward`: Rewards are based solely on DOM interactions.
    - `vision_reward`: Rewards are derived from vision-based interactions only.
    - `no_global_reward`: No global rewards are calculated.

- `--index`: Decide which data index to start with.
  - Type: String
  - Default: `-1`
  - Description: Use this parameter to specify a range or specific index for data processing. For example, `0,5` will process data from index 0 to 5.

- `--single_task_name`: Defines the task name of the single task to execute.
  - Type: String
  - Default: `"Find Dota 2 game and add all DLC to cart in steam."`
  - Description: Use this parameter to specify the task that the agent should perform.

#### Interaction Mode

Evaluating web agents in an online environment can sometimes be painful due to issues like network problems or bot tests on certain websites. Adopting an evaluation method that accommodates these issues allows for an accurate assessment of an agent's performance under specific current conditions. Additionally, we provide a more flexible interaction mode, enabling users to manually solve environmental issues and get the optimized performance of their web agents. You can simply set the `interaction_mode` parameter in `configs/setting.toml` to enable this feature. We will accumulate our implementation on error handling in online agent inference, and try to minimize human efforts by triggering only when exceptions occur in the following version. 

### Upload the Result for a Challenge

IMPORTANT: You should upload the generated out.json file to participate a challenge. To upload your result, use the following command:

```bash
python data/dataset_io.py upload \
    --file-path /path/to/your/file \
    --challenge-id your_challenge_id \
    --name your_agent_name \
    --base-model your_agent_base_model
```

Replace the placeholders with your actual values:

- `/path/to/your/file`: The path to the result you want to upload.
- `your_challenge_id`: The ID of the challenge you want to participate.
- `your_agent_name`: The agent name for the upload.
- `your_agent_base_model`: The agent base model information for the upload.

You can also submit through our platform. We will conduct an official check on your submission to prevent cheating.

## Create Your Own Benchmark Dataset

You can follow instructions on this [documentation](https://webcanvas.gitbook.io/webcanvas-docs) about how to create your own challenging benchmark for web agents. 

## 🤝 Contributing

We welcome contributions to WebCanvas!

In the coming updates, we will provide detailed guidelines on how to contribute to our project. This will include instructions on our coding standards, the process for submitting pull requests, and how to report issues, and more. Stay tuned for more information!

Thank you for your interest in improving WebCanvas. Your contributions are greatly appreciated and essential to the growth and success of our project.


## 🌐 Community

We are building a vibrant and inclusive community around WebCanvas! Join our community to stay up-to-date with the latest developments and to contribute to the project:

- [GitHub Discussions](https://github.com/iMeanAI/WebCanvas/discussions)
- [Discord](https://discord.gg/dhtgvJ52)

## 📢 Feedback

We value your feedback and suggestions!

We will be providing a detailed guide on how to give feedback in the upcoming documentation. This will include information on how to submit feedback, the types of feedback we are looking for, and how we plan to address and incorporate your suggestions. Stay tuned for more updates!

- [Talk to Founder](https://calendly.com/dehan/30min), we welcome any discussion and feedback on the future of live agent evaluation!

## Citation

If you use this project in your research, please cite our paper:

@article{pan2024webcanvas,
  title={WebCanvas: Benchmarking Web Agents in Online Environments},
  author={Pan, Yichen and Kong, Dehan and Zhou, Sida and Cui, Cheng and Leng, Yifei and Jiang, Bing and Liu, Hangyu and Shang, Yanyi and Zhou, Shuyan and Wu, Tongshuang and others},
  journal={arXiv preprint arXiv:2406.12373},
  year={2024}
}

#### References
[^1]: Deng, Xiang, et al. "Mind2web: Towards a generalist agent for the web." Advances in Neural Information Processing Systems 36 (2024).
[^2]: Zhou, Shuyan, et al. "Webarena: A realistic web environment for building autonomous agents." arXiv preprint arXiv:2307.13854 (2023).
[^3]: Mialon, Grégoire, et al. "Gaia: a benchmark for general ai assistants." arXiv preprint arXiv:2311.12983 (2023).
[^4]: Drouin, Alexandre, et al. "WorkArena: How Capable are Web Agents at Solving Common Knowledge Work Tasks?." arXiv preprint arXiv:2403.07718 (2024).