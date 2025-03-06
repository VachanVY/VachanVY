Hi, I'm Vachan! **I'm like Deep Learning and Systems Programming**

<img src="https://quotes-github-readme.vercel.app/api?type=horizontal&theme=dark" /><be>
<img src="https://github.com/user-attachments/assets/e173e76b-22a4-46dd-ad09-2b65ab9ed605" alt="Description" width="300">

# Projects:
## [**NeuroForge**](https://github.com/VachanVY/NeuroForge):
* Implemented Neural Network (Forward and Backward Propagation), Batchnorm and Layernorm, Dropout from scratch just using basic tensor methods
* [Neural Networks](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#neural-networks) => [*nn.ipynb*](https://github.com/VachanVY/NeuroForge/blob/main/nn.ipynb)
  * [Logistic Regression](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#logistic-regression)
  * [MLP](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#logistic-regression)
    * [Forward Propagation (Explained on Pen and Paper)](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#logistic-regression)
    * [Back Propagation (Equations Derived on Pen and Paper)](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#logistic-regression)
    * [Gradient Descent](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#logistic-regression)
    * [Train Loop](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#logistic-regression)
    * [Results](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#results)
* [Batch-Normalization and Layer-Normalization: **Why When Where & How?**](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#results) => [*batchnorm.ipynb*]( https://github.com/VachanVY/NeuroForge/blob/main/batchnorm.ipynb), [*layernorm.ipynb*](https://github.com/VachanVY/NeuroForge/blob/main/layernorm.ipynb)
  * [Batch-Normalization](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#batch-normalization)
  * [Layer-Normalization](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#layer-normalization)
  * [Comparision](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#comparision)
* [Dropout: **Why When Where & How?**](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#dropout-paper-deep-learning-book) => [*dropout.ipynb*](https://github.com/VachanVY/NeuroForge/blob/main/dropout.ipynb), [*dropout_scale.ipynb*](https://github.com/VachanVY/NeuroForge/blob/main/dropout_scale.ipynb)
  * [Comparision before and after scaling the model](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#comparision-1) => [*dropout_scale.ipynb*](https://github.com/VachanVY/NeuroForge/blob/main/dropout_scale.ipynb), [nn_scale.ipynb](https://github.com/VachanVY/NeuroForge/blob/main/nn_scale.ipynb)
* [Adam and AdamW](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#adam-and-adamw-adam-with-weight-decay-optimizers)
  * [Adam](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#adam-and-adamw-adam-with-weight-decay-optimizers)
  * [AdamW](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#adam-and-adamw-adam-with-weight-decay-optimizers)
* [Model Distillation](https://github.com/VachanVY/NeuroForge/tree/main?tab=readme-ov-file#model-distillation-paper-reference) => [*distillation.ipynb*](https://github.com/VachanVY/NeuroForge/blob/main/distillation.ipynb)
* [Mixture-Of-Experts (MoE) Layers](https://github.com/VachanVY/NeuroForge?tab=readme-ov-file#mixture-of-experts-moe-layer)


## Transformers
```mermaid
graph TD;
    Transformers -->|Text| GPT;
    Transformers -->|Images| Vision_Transformers["Vision Transformers"];
    Transformers -->|Audio| MAGNeT["MAGNeT"];
    Transformers --> |Video| Video_Vision_Transformers["Video Vision Transformers"];
    Transformers -->|Diffusion| Diffusion_Transformers["Diffusion Transformers"];

    GPT --> Multi_Modal_Transformers["Multi-Modal Transformer"];
    Vision_Transformers --> Multi_Modal_Transformers;
    MAGNeT --> Multi_Modal_Transformers;
    Video_Vision_Transformers --> Multi_Modal_Transformers;
    Diffusion_Transformers --> Multi_Modal_Transformers;

    Multi_Modal_Transformers --> LLMs["Large Language Models (LLMs)"];
    RLHF["Reinforcement Learning from Human Feedback (RLHF)"] --> LLMs;

    Reinforcement_Learning --> RLHF;

    LLMs --> Reasoning_LLMs["Reasoning LLMs"];
    Reinforcement_Learning --> Reasoning_LLMs;
```

### [**gpt.jax**](https://github.com/VachanVY/gpt.jax): 
* GPT written in `jax`, trained on `tiny shakespeare dataset (1.1 MB text data)` and scaled it on the `tiny stories dataset (~2 GB text data)`
  | Model-Params       |`d_model`| `n_heads`  | `maximum_context_length` | `num_layers`  | `vocab_size` | Estimated Validation Loss on tiny stories dataset   |
  | :-------------:    |:-------:|:----------:|:------------------------:|:--------------|:------------:|:-------------------------:|
  | *280K*             |   64    |     8      |           512            |       5       |      512     |      **1.33**             |
  | *15M*              |   288   |     6      |           256            |       6       |     32000    |      **1.19**             |
  | *45M*              |   512   |     8      |           1024           |       8       |     32000    |      **TODO**             |
  | *110M*             |   768   |     12     |           2048           |       12      |     32000    |      **TODO**             |
* Model: `15M` | Prompt: `Once upon a time,` | Sampling Technique: `Greedy sampling`
    ```
    Once upon a time, there was a little girl named Lily. She loved to play with her toys and eat yummy food. One day, she found a big, round thing in her room. It was a microscope. Lily was very curious about it.
    Lily wanted to see what was inside the microscope. She tried to open it, but it was very hard. She tried and tried, but she could not open it. Lily felt sad and wanted to find a way to open the microscope.
    Then, Lily had an idea. She asked her mom for help. Her mom showed her how to open the microscope. Lily was so happy! She looked through the microscope and saw many tiny things. She was so excited to see the tiny things. Lily and her mom had a fun day together.
    ```
* Prompt: `Once upon a time, in a big forest, there was a fearful little dog named Spot` | Sampling Technique: `Greedy sampling`
    ```
    Once upon a time, in a big forest, there was a fearful little dog named Spot. Spot was scared of many things. One day, Spot saw a big tree with a hole in it. He thought, "I want to see what is inside the hole."
    Spot went to the tree and looked inside the hole. He saw a little bird with a hurt wing. Spot said, "I will help you, little bird." He used his paw to gently lift the bird out of the hole. The bird was very happy and said, "Thank you, Spot!"
    Spot and the bird became good friends. They played together in the forest every day. Spot learned that it is good to help others, even if they are scared of something. And they lived happily ever after.
    ```

### [**Diffusion Transformers**](https://github.com/VachanVY/diffusion-transformer)
* [**CelebA**](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#celeba) 
   * **[More Generated-images](https://github.com/VachanVY/diffusion-transformer/tree/main/images)** <====== See more Model Generated Images here
   * **[Training-insights](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#training-insights)**
* **[MNIST-experiment](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#mnist-experiment)**
   * [**Training on MNIST**](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#training-on-mnist)
* **[Diffusion-Transformers Paper Summary](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#latent-diffusion-models)**
* Some generated images:\
    <img src="https://github.com/user-attachments/assets/6cbe6bc7-1e7a-44ed-83ae-df64c40bf8d4" alt="Alt text" width="300">
    <img src="https://github.com/user-attachments/assets/181f31fe-4b4c-4719-93df-f3d767853608" alt="Alt text" width="300">
    <img src="https://github.com/user-attachments/assets/eb463c13-45b2-474e-a431-b3b296fe00c4" alt="Alt text" width="300">
    <img src="https://github.com/user-attachments/assets/2048e0ba-4ebd-48f1-a379-df39680495f9" alt="Alt text" width="300">
    <img src="https://github.com/user-attachments/assets/e7bea3b5-ce52-4de6-a03e-a7694e66b320" alt="Alt text" width="300">
    <img src="https://github.com/user-attachments/assets/fe7d1f7d-f7a8-4059-a1f2-172a0e2345d5" alt="Alt text" width="300">

# [Reinforcement-Learning](https://github.com/VachanVY/Reinforcement-Learning)
| **Algorithms**                          | **Environment (Name & Goal)**               | **Environment GIF**                           | **Plots**               |
|----------------------------------------|---------------------------------------------|-----------------------------------------------|-------------------------------------------|
| [Policy Iteration](#policy-iteration)  | **Frozen Lake**: The player makes moves until they reach the goal or fall in a hole. The lake is slippery (unless disabled) so the player may move perpendicular to the intended direction sometimes.               | ![pol](images/frozen_lake_policy_iteration.gif)  ![pol](images/frozen_slippery_lake_policy_iteration.gif)   | - |
| [Value Iteration](#value-iteration)    | **Taxi-v3**: The taxi starts at a random location within the grid. The passenger starts at one of the designated pick-up locations. The passenger also has a randomly assigned destination (one of the four designated locations).                | ![Gridworld](images/Taxi-v3_value_iteration.gif) ![Gridworld](images/Taxi-v3_value_iteration1.gif) ![Gridworld](images/Taxi-v3_value_iteration2.gif)     | - |
| [Monte Carlo Exploring Starts](#monte-carlo-exploring-starts) | **Blackjack-v1**: a card game where the goal is to beat the dealer by obtaining cards that sum to closer to 21 (without going over 21) than the dealer's cards        | ![Blackjack](images/mc_blackjack_value_iteration1.gif)  | ![Graph](images/blackjack_actions.png) ![Graph](images/blackjack_qvals.png) |
| [Sarsa](#sarsa)                        | **CliffWalking-v0**: Reach goal without falling  | ![CliffWalking](images/cliff_walking_sarsa.gif)           | ![Graph](images/cliff_walking_gamma0.99_alpha0.1_epsilon0.1.png)  Sarsa: Orange       |
| [Q-learning](#q-learning)              | **CliffWalking-v0**: Reach goal without falling  | ![CliffWalking](images/cliff_walking_qlearning.gif)      | ![Graph](images/cliff_walking_gamma0.99_alpha0.1_epsilon0.1.png)  Q-learning: Blue    |
| [Expected Sarsa](#expected-sarsa)      | **CliffWalking-v0**: Reach goal without falling  | ![CliffWalking](images/cliff_walking_expected_sarsa.gif)  | ![Graph](images/cliff_walking_gamma0.99_alpha0.1_epsilon0.1.png)  Expected Sarsa: Green |
| [Double Q-learning](#double-q-learning)          | **CliffWalking-v0**: Reach goal without falling  | ![CliffWalking](images/cliff_walking_double_qlearning.gif)  | ![Graph](images/cliff_walking_gamma0.99_alpha0.1_epsilon0.1.png) Double Q-learning: Red |
| n-step Bootstrapping **(TODO)**        | -                                           | -                                             | -                                         |
| [Dyna-Q]() | **ShortcutMazeEnv** (*custom made env*): Reach the goal dodging obstacles | ![maze0](images/shortcut_maze_before_Dyna-Q_with_25_planning_steps.gif) ![maze](images/shortcut_maze_after_Dyna-Q_with_25_planning_steps.gif) | ![compare by steps](images/dyna_q_num_planning_steps_zoomed.png)                                         | 
| [Prioritized Sweeping]() | **ShortcutMazeEnv** (*custom made env*): Reach the goal dodging obstacles | ![maze0](images/shortcut_maze_prioritized_sweeping_maze_env.gif) | ![steps](images/prioritized_sweeping_maze_env_num_training_curves.png) ![sum rewards](images/prioritized_sweeping_maze_env_sum_rewards.png)                                        | 
| [Monte-Carlo Policy-Gradient](#monte-carlo-policy-gradient) | **CartPole-v1**: goal is to balance the pole by applying forces in the left and right direction on the cart.                 | ![CartPole](images/actor_critic_cartpole.gif)               | ![Graph](images/monte_carlo_policy_gradient_cartpole_train_graph.png)         |
| [REINFORCE with Baseline](#reinforce-with-baseline) | **CartPole-v1**: goal is to balance the pole by applying forces in the left and right direction on the cart.                 | ![CartPole](images/to/reinforce_baseline.gif)  | - |
| [One-Step Actor-Critic](#one-step-actor-critic) | **CartPole-v1**: goal is to balance the pole by applying forces in the left and right direction on the cart.                 | ![CartPole](images/actor_critic_cartpole.gif)        | ![Graph](images/actor_critic_cartpole_rewards.png) |
| Policy Gradient on Continuous Actions **(TODO)** | -                                    | -                                             | -                                         |
| On-policy Control with Approximation **(TODO)** | -                                   | -                                             | -                                         |
| Off-policy Methods with Approximation **(TODO)** | -                                   | -                                             | -                                         |
| Eligibility Traces **(TODO)**          | -                                           | -                                             | -                                         |

---

## Deep Reinforcement Learning: Paper Implementations

| **Year** | **Paper**                                                       | **Environment (Name & Goal)**               | **Environment GIF**                           | **Plots**               |
|----------|-----------------------------------------------------------------|---------------------------------------------|-----------------------------------------------|-------------------------|
| 2013     | [Playing Atari with Deep Reinforcement Learning](#)             | **ALE/Pong-v5** - You control the right paddle, you compete against the left paddle controlled by the computer. You each try to keep deflecting the ball away from your goal and into your opponent’s goal.   | <img src="images/dqn_pong.gif" width="200">                 | <img src="images/loss_dqn.png" width="1000"> <img src="images/Sum_of_Reward.svg" width="1000"> <img src="images/Steps_per_Episode.svg" width="1000"> |
| 2014     | [Deep Deterministic Policy Gradient (DDPG)](#)                  | **Pendulum-v1** - The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an upright position, with its center of gravity right above the fixed point.     | ![Pendulum](images/ddpg_pendulum.gif)                | ![Plot](images/ddpg_on_Pendulum-v1.png) |
| 2015, 2016     | [Deep Reinforcement Learning with Double Q-Learning + Prioritized Experience Replay](#)         | -                   | -                   | - |
| 2017     | [Proximal Policy Optimization (PPO)](#)                         | **LunarLander-v3**: This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off                 | ![opaos](images/lunar_lander.gif)                 | ![Plot](images/LunarLander-v3_rewards.png) |
| 2018     | [Soft Actor-Critic (SAC)](#)                                    | **InvertedDoublePendulum-v5**: The cart can be pushed left or right, and the goal is to balance the second pole on top of the first pole, which is in turn on top of the cart, by applying continuous forces to the cart. | Constant Alpha: ![Plot](images/sac_inverteddoublependulum-v5_.gif) Learnable Alpha (**TODO**: add an explanation for adaptive alpha loss): ![Plot](images/sac_adaptive_alpha_inverteddoublependulum-v5_.gif) | Constant Alpha: ![Plot](images/sac_rewards_InvertedDoublePendulum-v5.png) Learnable Alpha: ![Plot](images/sac_rewards_adaptive_alpha_InvertedDoublePendulum-v5.png) |
| 2017     | [Mastering the Game of Go without Human Knowledge](#)           | Go - Win against self-played adversary       | -                   |  -   |
| 2017     | [AlphaZero](#)                                                  | Chess - Beat traditional engines             | -             |  -   |
| 2020     | [Mastering Atari, Go, Chess and Shogi with a Learned Model](#)  | Multiple Environments (Planning with Models)| -                | ! -   |
| 20xx     | [AlphaFold](#)                                                  | Protein Folding - Predict protein structures| -     | -   |



<!-- ### [**ViViT**](https://github.com/VachanVY/ViVIT):
* Video Vision Transformer in PyTorch
* Test trained on MNIST images by stacking images of the same digit in the time dimension
* TODO: Scale the model and train it on a proper large dataset...

### [**Vision-Transformers**](https://github.com/VachanVY/Vision-Transformers):
* Vision Transformers in `jax`, trained on `MNIST` dataset
* TODO: Scale ViT and train on a larger dataset -->

<!-- 
### **Mugen**
* Going to make a website for music generation using *Pytorch* only. On-going project...
* TODO: Test train it on the MusicBench dataset, takes 5 seconds/step on my GPU, very slow... need GPUs
* TODO: Scale on large lyrical music datasets
* Repeat for the below models
* ### Models for this Project
  * - [x] [Non Autoregressive Transformer](https://github.com/VachanVY/MAGNeT)
  * - [ ] Autoregressive Transformer
  * - [ ] Diffusion Transformer
-->
<!--
# [Reinforcement-Learning](https://github.com/VachanVY/Reinforcement-Learning)
> Below links don't redirect anywhere, gotta refactor the code and add links, for now go to the repo directly👆
## Reinforcement Learning: An Introduction by Andrew Barto and Richard S. Sutton
* [Dynamic Programming]()
  * [Policy Iteration - Policy Evaluation & Policy Iteration]()
  * [Value Iteration]()
* [Monte-Carlo Methods]()
  * [Monte Carlo Exploring Starts]()
* [Temporal-Difference (Tabular)]()
  * [Sarsa]()
  * [Q-learning]()
  * [Expected Sarsa]()
  * [Double Q-learning]()
* n-step Bootstrapping
* Planning and Learning with Tabular Methods
* [On-policy Prediction with Approximation]()
  * Covered in [Papers]() Section, where we use function approximators like Neural Networks for RL
* On-policy Control with Approximation
* Off-policy Methods with Approximation
* Eligibility Traces
* [Policy Gradient Methods]()
  * [Monte-Carlo Policy-Gradient]()
  * [REINFORCE with Baseline]()
  * [One-Step Actor-Critic]()
  * Policy Gradient on Continuous Actions

---
## Reinforcement Learning: Paper Implementations
* [2013: Playing Atari with Deep Reinforcement Learning]()
* Prioritized DDQN || 2015: Deep Reinforcement Learning with Double Q-learning **+** 2016 Prioritized Experience Replay || **(TODO)**
* [2017: Proximal Policy Optimization (PPO)]()
* [2014: Deterministic Policy Gradient]()
* [2018: Soft Actor-Critic]()
* AlphaGo, AlphaZero, AlphaFold, etc:
  * 2017: Mastering the game of go without human knowledge
  * 2017: AlphaZero
  * 2020: Mastering Atari, Go, chess and shogi by planning with a learned model
  * 20xx: AlphaFold
* (many more to be added...)
---

### [Transfusion (A Multi-Modal Transformer)](https://github.com/VachanVY/Transfusion.torch)
* Transfusion is a Multi-Modal Transformer, it can generate text like GPTs and images like Diffusion Models, all at once in one go not separately!
* It can easily switch between text and image modalities for generations, and it is nothing complicated, just a single transformer with some modality-specific components!
* This can easily be extended to other modalities like videos, audio, etc, but for now, it can only take images and text as input
* **`TODO`: Train on a large Multi-Modal Dataset (something like tiny stories dataset with images in between illustrating the story...?)**
-->

<img  src="https://github-profile-trophy.vercel.app/?username=VachanVY&theme=gruvbox&row=1&column=6&no-frame=true&no-bg=true" /><br>

<img height="137px" src="https://github-readme-stats-git-masterrstaa-rickstaa.vercel.app/api?username=VachanVY&hide_title=true&hide_border=true&show_icons=trueline_height=21&text_color=000&icon_color=000&bg_color=0,ea6161,ffc64d,fffc4d,52fa5a&theme=graywhite" />

<img height="137px" src="https://github-readme-stats-git-masterrstaa-rickstaa.vercel.app/api/top-langs/?username=VachanVY&hide_title=true&hide_border=true&layout=compact&langs_count=6&text_color=000&icon_color=fff&bg_color=0,52fa5a,4dfcff,c64dff&theme=graywhite" /><br><br>

