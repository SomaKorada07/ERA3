# Data Preparation

- We created two distinct datasets: a step-wise prompt-completion dataset and a conversational role-based dataset.

## Step-Wise Prompt-Completion Dataset
- We leveraged existing agent trajectories from AppWorld (sampled traces of user tasks) and decomposed them into steps.
- Each data point consists of a prompt and completion pair:
- **Prompt**:
    - For each step in the trajectory, the prompt was structured to include:
        - user_instruction: the original task or goal given by the user.
        - thoughts_so_far: a description of the reasoning or thought process leading to the current step.
        - latest_observation: the system’s most recent output or change in environment.
    - The intent of the prompt is to instruct the model to generate what the agent should "think" and "do" next.
 
### Prompt Template:
```
You are an intelligent AI Assistant whose job is to achieve the user's instruction completely autonomously.
To do this, you will need to interact with app/s (e.g. venmo) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.
 
Here are three key APIs that you need to know to get more information
 
# To get a list of apps that are available to you.
apis.api_docs.show_app_descriptions()
 
# To get the list of apis under any app listed above, e.g. supervisor
apis.api_docs.show_api_descriptions(app_name='supervisor')
 
# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords')
User Instruction : 
```{}```
The following list contains the ordered history of actions already taken and the corresponding outputs. The last element of the list represents the latest action and its observation: 
```{}```
 
Based on User instruction, history of steps taken and outputs, decide your next step.
Respond with the action to take and the thought behind it.
 
Respond in the following json format:
{{"thought" : "...",
"action" : "..."}}
```
- **Completion**:
    - The target (completion) for each prompt includes:
        - next_thought: the rationale or intention for the next move.
        - next_action: the actual executable step to be taken by the agent.

### Data Generation using GPT-4o:
- We used GPT-4o to synthesize 200 additional user queries, mirroring the style and intent of existing examples.
- For each synthetic query, we generated full step-wise prompt-completion trajectories, using the same format as manually constructed examples.
- Since data generation is a one-time task at the start of the model development pipeline, we prioritized quality over cost. GPT-4o was chosen over open-source models due to its superior reasoning, coherence, and contextual relevance in generating high-quality structured examples.
- This augmentation allowed coverage across multiple intents, including send_money, request_payment, and add_balance.


## Conversational Dataset (Role-Based Messages)
- We simulated a chat-like format between a user and the assistant agent, focusing only on the send_money intent.

### Synthetic Query Generation:
- Using a custom generation script, we created 5000+ diverse user utterances simulating real-world money transfer scenarios. This was done by randomly selecting from different options of entities and fitting them into various phrases for the send_money intent. Linguistic variety (e.g., "Shoot over 50 to Mike", "Can you zelle 30 bucks to Priya for gas?") was also incorporated, helping the model generalize better to colloquial forms. These included variation in:
    - Amounts (e.g., "$50", "a tenner", "75 bucks")
    - Recipients
    - Purposes (e.g., "lunch", "Uber", "concert tickets")

### Trajectory Templates:
- For each user message, we created a conversation trajectory using templates structured as:
```messages = [
 {"role": "user", "content": "<user request>"},
 {"role": "assistant", "content": "<acknowledgement or follow-up>"},
 {"role": "user", "content": "<answer or confirmation>"},
 ...
 ```


# Model Fine Tuning
## SFT Approach

### Approach 1
- We did full fine tuning using **Qwen 0.5B Instruct model** using the dataset (Prompt-Completion format) with 1595 records ([simulated_tasks_train.jsonl](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Data/simulated_tasks_train.jsonl)) - refer to [Starter.ipynb](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Notebooks/Starter.ipynb).

### Approach 2
- We did **PEFT with LORA** using **Qwen 14B Instruct model** using the dataset (Prompt-Completion format) with 1595 records with a change in prompt ([simulated_tasks_train_updated.csv](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Data/simulated_tasks_train_updated.csv)) - refer to [Training_2.ipynb](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Notebooks/Training_2.ipynb).
- BitsAndBytes Config is
```
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```
- LORA Config is
```
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```
- SFT Config is
```
training_args = SFTConfig(
    output_dir="./14b_results",  # Directory to save the model and checkpoints
    num_train_epochs=30,      # Number of training epochs
    per_device_train_batch_size=4, # Batch size per GPU/CPU
    optim="paged_adamw_8bit",
    fp16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    lr_scheduler_type='cosine',
    eos_token="<|im_end|>"
)
```

### Approach 3
- We repeated approach 2 on a different dataset - conversational dataset with focus on only P2P intent with ~70k training records - refer to [02_Training.ipynb](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Notebooks/02_Training.ipynb).
- We fine tuned Qwen 14B model for just one epoch.


## GRPO Approach
- We attempted to use GRPO on Qwen 14B Instruct model using the dataset (Prompt-Completion format) with 1595 records (simulated_tasks_train.jsonl) - refer to [qwen_RL_GRPO.ipynb](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Notebooks/qwen_RL_GRPO.ipynb).
- Reward system is designed to execute the model completion (code) and based on the code execution result the reward would be assigned - if code execution is erroring out give -1 reward and if successful give +1 reward. Basically, iterate through the total number of interactions or until task is completed, to reward in this fashion and get a total reward for the task.
- Not much analysis has been done on this approach yet and is in our TO-DO list.


# Evaluation
- We primarily used **AppWorld** for the evaluation along with **minimal-react-agent** framework provided by AppWorld.
- We evaluated Qwen 14B Instruct base model and also evaluated using GPT 4o model - refer to [Base_Inferencing.ipynb](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Notebooks/Base_Inferencing.ipynb).
- We evaluated our fine tuned model - refer to [FT_Inferencing.ipynb](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Notebooks/FT_Inferencing.ipynb).
- We served the models using **vLLM** for inferencing.


# Results
- We see the **pass percentage** of a few tasks has **increased** with our fine tuned model compared with the base model showing at par performance of GPT 4o model - refer to [inferencing_14b_test_normal_updated_23rdJun.csv](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Inferencing_Results/inferencing_14b_test_normal_updated_23rdJun.csv).
- Below are the pass percentages across different models (the below list _shows only some tasks and NOT all the tasks_ which show improvement in pass percentage):

| Index | Task ID    | Pass Percentage for Qwen 14B Base Model (%) | Pass Percentage for GPT 4o Model (%) | Pass Percentage for Fine Tuned Model (%) |
|:-----:|:----------:|:-------------------------------------------:|:------------------------------------:|:----------------------------------------:|
| 8     | 325d6ec_2  | 0.0                                         | 20.0                                  | 20.0                                      |
| 9     | 325d6ec_3  | 0.0                                         | 20.0                                  | 20.0                                      |
| 16    | 634f342_1  | 0.0                                         | 14.3                                  | 14.3                                      |
| 17    | 634f342_2  | 0.0                                         | 14.3                                  | 14.3                                      |
| 19    | 8749218_1  | 33.3                                        | 66.7                                 | 66.7                                     |
| 21    | 8749218_3  | 33.3                                        | 66.7                                 | 66.7                                     |
| 22    | 2d9f728_1  | 10.0                                        | 20.0                                 | 20.0                                     |
| 76    | 652485c_1  | 9.1                                         | 18.2                                  | 18.2                                      |
| 77    | 652485c_2  | 9.1                                         | 18.2                                  | 18.2                                      |
| 78    | 652485c_3  | 9.1                                         | 18.2                                  | 18.2                                      |
| 79    | ccf4b82_1  | 18.2                                        | 27.3                                 | 27.3                                     |
| 80    | ccf4b82_2  | 18.2                                        | 27.3                                 | 27.3                                     |
| 81    | ccf4b82_3  | 18.2                                        | 27.3                                 | 27.3                                     |
| 124   | b6d1104_1  | 10.0                                        | 20.0                                 | 20.0                                     |
| 125   | b6d1104_2  | 10.0                                        | 40.0                                 | 40.0                                     |
| 126   | b6d1104_3  | 10.0                                        | 40.0                                 | 40.0                                     |

 

# Challenges Faced
- We were not successful in doing evaluation using Qwen 32B Instruct model, were getting the following error - 

![error](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Screenshots/32B_error.png "error")

- We were not able to evaluate using ```appworld run``` command to get the TGC and SGC scores.



# Future Steps
- Continue our exploration on GRPO. - 3-4 days of effort
- Focus on ONLY P2P intent and build more data around it. - 1-2 days of effort
- Do SFT on improved dataset given more GPUs and better base model (Qwen 32B Instruct model) - 1 day of effort
- Continue building a robust agentic framework which includes multi agents - **Perception Agent (Critic)**, **Decision Agent (Actor)** and more. - 2-3 days of effort
    - Plan to use our fine tuned model for the **Decision Agent**.

    ![Perception Output](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Screenshots/Perception_Output.png "Perception Output")
    
    ![Decision Graph](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Screenshots/Decision_Graph.png "Decision Graph")
    
    ![Decision Code Execution](https://github.com/SomaKorada07/ERA3/blob/main/CapstoneProject/Screenshots/Decision_Code_Execution.png "Decision Code Execution")