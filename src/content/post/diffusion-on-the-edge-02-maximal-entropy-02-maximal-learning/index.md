---
title: 'Diffusion on the edge - Part II: Maximal manifold exploration'
publishDate: 2025-11-02
frontSlug: diffusion-on-the-edge-02-maximal-entropy-02-maximal-learning
toc_items:
- level: 1
  text: 'Diffusion on the edge - Part II: Maximal manifold exploration'
  id: diffusion-on-the-edge-part-ii-maximal-manifold-exploration
- level: 2
  text: Main concept
  id: main-concept
- level: 2
  text: Dataset
  id: dataset
- level: 2
  text: Transformations
  id: transformations
- level: 1
  text: Modeling
  id: modeling
- level: 3
  text: Backwards sampling
  id: backwards-sampling
- level: 3
  text: Original model conclusion
  id: original-model-conclusion
- level: 1
  text: Maximal manifold exploration
  id: maximal-manifold-exploration
- level: 2
  text: Maximal exploration formalism
  id: maximal-exploration-formalism
- level: 2
  text: The first variation and surprise
  id: the-first-variation-and-surprise
- level: 2
  text: Implementation of manifold exploration in practice
  id: implementation-of-manifold-exploration-in-practice
- level: 2
  text: Fine-tuned model sampling
  id: fine-tuned-model-sampling
- level: 3
  text: Visualization
  id: visualization
- level: 3
  text: Qualitative values
  id: qualitative-values
- level: 2
  text: Density approximation
  id: density-approximation
- level: 1
  text: Conclusion
  id: conclusion
prev:
  title: Diffusion models - Introduction
  url: /blog/diffusion-on-the-edge-01-introduction-notebook
---


# Diffusion on the edge - Part II: Maximal manifold exploration {#diffusion-on-the-edge-part-ii-maximal-manifold-exploration}

In the previous part, we explored the basics of score-matching based diffusion models and continous-time stochastic processes.
This follow up piece will walk through the basic ideas explored in the paper _Provable Maximum Entropy Manifold Exploration via Diffusion Models_ by De Santi et al.

## Main concept {#main-concept}

The paper introduces the concept of exploring the data manifold maximally by fine tuning a pre-learned diffusion model. Mathematically, the paper describes how diffusion models can miss some of the underlying data distribution. Our goal is to implement the described S-MEME algorithm and generate samples from low-density regions of the data manifold.

## Dataset {#dataset}

For simplicity, as we develop this implementation code we should have a simple and robust dataset on hand.

Let's take as our fundamental data manifold all possible 2D-triangles based on distance. So each data point is a triplet of numbers, $(a, b, c)$ where the measurements $a, b, c$ are distances between points $A, B, C$ defined as $a = |AB|, b = |BC|, c = |CA|$. All of these points lie on some non-trivial submanifold of $\mathbb{R}^3$. For simplicity, let's also constrict the distances to be less than $1$. Therefore, the possible data points lie as a subset of the unit cube, $(a, b, c) \in [0, 1]^3$. Furthermore, let always be true that $a \leq b \leq c$, so we maintain an order on the sides.

## Transformations {#transformations}

We will utilize some transformations for the raw side lengths

- Side normalized $(a, b, c) = (a / c, b / c, 1)$
- Perimeter normalized $(a, b, c) = (a / l, b / l, c / l)$ where $l = a + b + c$

For training neural networks, we do a mean and standard deviation scaling as well, on top of these two normalization approaches.


```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns

from diffusion_on_the_edge.core.grid import TimeGrid
from diffusion_on_the_edge.processes import OUTorchParams, OUTorch
from diffusion_on_the_edge.data.ou_dataset import OUDiffusionDatasetVectorized
from diffusion_on_the_edge.data.triangle_dataset import DatasetOptions, generate_triangle_dataset, dataset_from_sides
from diffusion_on_the_edge.modeling.model import SimpleScoreNet
from diffusion_on_the_edge.modeling.training import train_scorenet, TrainConfig
from diffusion_on_the_edge.sde.torch_backend import reverse_pc_sampler_torch
from diffusion_on_the_edge.modeling.tuning_model import build_teacher_student

ASSETS_FOLDER = './assets'
```


```python
# Dataset sizes
B, D = 1024, 3
ou_params = OUTorchParams(dim = 4, theta = 0.3, sigma = 0.5)
grid = TimeGrid(t0 = 0, t1 = 1, N = 101)
dataset_params = DatasetOptions(
    n_samples=B,
    side_bias={'equilateral': 0.00, 'isosceles': 0.05, 'scalene': 0.95},
    angle_bias={'right': 0.00, 'acute': 0.90, 'obtuse': 0.1},
)
```


```python
# Generate dataset with both side and angle biases
dataset_df = generate_triangle_dataset(dataset_params)
dataset_df.head()
```
<pre><code class="language-output">a         b         c side_type angle_type  angle_A_deg  \
0  0.376802  0.390557  0.534150   scalene      acute    44.835878   
1  0.188496  0.494622  0.496998   scalene      acute    21.914174   
2  0.573584  0.795410  0.957273   scalene      acute    36.758696   
3  0.801615  0.821972  0.932127   scalene      acute    53.944037   
4  0.713944  0.818943  0.880056   scalene      acute    49.535806   

   angle_B_deg  angle_C_deg  perimeter      area  a_over_c  b_over_c  \
0    46.954934    88.209188   1.301509  0.073545  0.705423  0.731174   
1    78.333841    79.751985   1.180116  0.045873  0.379268  0.995219   
2    56.087156    87.154148   2.326267  0.227836  0.599185  0.830912   
3    55.993345    70.062617   2.555715  0.309707  0.859985  0.881824   
4    60.774273    69.689922   2.412943  0.274164  0.811249  0.930558   

          x         y  a_over_l  b_over_l  c_over_l  
0  0.257195  0.275373  0.289512  0.300080  0.410408  
1  0.038115  0.184602  0.159726  0.419130  0.421144  
2  0.320020  0.476010  0.246568  0.341925  0.411506  
3  0.448335  0.664517  0.313656  0.321621  0.364723  
4  0.348584  0.623061  0.295881  0.339396  0.364723</code></pre>
Next, we can take a look at our generated dataset of triangles.


```python
import matplotlib.pyplot as plt

def plot_triangle(a, b, c, ax, label=None, color='blue'):
    """
    Plot a triangle in 2D given side lengths a, b, c.
    Triangle ABC with:
        A at (0, 0),
        B at (c, 0),
        C determined by cosine law from a, b, c
    """
    # Fixed base AB
    A = np.array([0, 0])
    B = np.array([c, 0])
    
    # Use law of cosines to find angle at C
    cos_angle = (a**2 + c**2 - b**2) / (2 * a * c)
    cos_angle = np.clip(cos_angle, -1, 1)  # numerical safety
    angle = np.arccos(cos_angle)
    # Coordinates of C using polar transformation from A
    C = np.array([a * np.cos(angle), a * np.sin(angle)])
    triangle = np.array([A, B, C, A])
    ax.plot(triangle[:, 0], triangle[:, 1], marker='o', color=color, label=label, alpha = 0.7)
    ax.set_aspect('equal')

def visualize_sample_triangles(df, n=5):
    """
    Visualize n triangles from the dataframe.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sample: pd.DataFrame = df.sample(n=n).reset_index()
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    
    for i, row in sample.iterrows():
        a, b, c = row.a, row.b, row.c # Assumes first three are the sides
        plot_triangle(a, b, c, ax, label=f"{row.side_type}, {row.angle_type}", color=colors[i])
    ax.legend()
    return fig, ax
```


```python
# Visualize a few triangles
fig, ax = visualize_sample_triangles(dataset_df, n=5)
ax.grid()
ax.set_title('Triangles in 2-dimensions')
plt.tight_layout()
fig.savefig(f'{ASSETS_FOLDER}/triangle_plot.png')
plt.close()
```

<img src="/blog/diffusion-on-the-edge-02-maximal-entropy-02-maximal-learning/assets/triangle-plot.60fc6eb9.png" width=720 height=360 />

Our dataset contains normalized side lengths as well, below is a plot of the perimeter scaled versions.


```python
# Plotting perimeter normalized triangles
plotting_df = dataset_df[['a_over_l', 'b_over_l', 'c_over_l', 'side_type', 'angle_type']].rename({'a_over_l': 'a', 'b_over_l': 'b', 'c_over_l': 'c'}, axis = 1)
fig, ax = visualize_sample_triangles(plotting_df)
ax.grid()
ax.set_title('Perimeter scaled triangles')
plt.savefig(f"{ASSETS_FOLDER}/perimeter_scaled_tri.png")
plt.close()
```

<img src="/blog/diffusion-on-the-edge-02-maximal-entropy-02-maximal-learning/assets/perimeter-scaled-tri.8fccc591.png" width=720 height=360>

# Modeling {#modeling}

We create a $4$-dimensional OU diffusion process with specific $\lambda$ and $\sigma$ parameters, as described in the first part. Alongside it, we create a neural network model to be our first training model.


```python
ou_process = OUTorch(ou_params)
model = SimpleScoreNet(input_dimension=4, layer_count=4, hidden_dim=256)
```

Now we define an iterable dataset for the diffusion model, the dataset contains tuples $(X, t, \nabla_x p_t(x))$.

In other words, the iterable dataset generates samples $x \sim X_t$ from the diffusion process from the starting set of triangles. Alonside the samples, we compute the correct value for the gradient of the density $p_t(x)$. We use the normalized side lengths and provide the perimeter as an additional feature $X = (a / l, b / l, c / l, l)$. After the perimeter normalization, we also compute the mean and standard deviation, and use them as a second normalization step for the network.


```python
scaled_dataset = dataset_df[['a_over_l', 'b_over_l', 'c_over_l', 'perimeter']].to_numpy()
scaled_dataset_torch = torch.from_numpy(scaled_dataset)
means = scaled_dataset_torch.mean(dim = 0)
stds = scaled_dataset_torch.std(dim = 0)
scaled_dataset_torch = (scaled_dataset_torch - means) / stds

dataset = OUDiffusionDatasetVectorized(
    ou_process,
    x0_pool=scaled_dataset_torch,
    T_max=grid.t1,
    batch_size=32,
    batches_per_epoch=100
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
```


```python
# Train the scoring model on the triangle dataset
config = TrainConfig(epochs=16, lr=0.001)
train_result = train_scorenet(model, dataloader, cfg=config)
score_model = train_result.get('model').to('cpu').eval()
training_losses = train_result.get('losses')
```


```python
fig = sns.lineplot(y = training_losses, x = range(1, config.epochs + 1)).set(xlim=(1, config.epochs + 1), xlabel='Epoch', ylabel='Training loss')
plt.grid()
plt.savefig(f'{ASSETS_FOLDER}/training_loss.png')
plt.close()
```

<img src="/blog/diffusion-on-the-edge-02-maximal-entropy-02-maximal-learning/assets/training-loss.8588eb88.png" width=560 height=400 />

Here we have plotted the training loss of our triangle diffusion model, it exhibits nice inverse exponential loss.

### Backwards sampling {#backwards-sampling}

Now we have the model, we can sample backwards as in the introduction post.


```python
# Generate random samples
x_T = torch.randn(B, D + 1).to('cpu')

# Generate new samples using backwards SDE
samples = reverse_pc_sampler_torch(
    x_T=x_T,
    process=ou_process,
    score_model=score_model,
    grid = grid,
    snr=0.15,
)

# Transform back from perimeter scaled values
transformed_samples = samples * stds + means
samples_arr = transformed_samples.detach().cpu().numpy()
samples_arr = samples_arr[:, :3] * samples_arr[:, 3].reshape(-1, 1)
```


```python
generation_options = DatasetOptions(include_geometry=True, include_normalized_sides=True, include_perimeter_normalized_sides=False, include_planar_embedding=False)
generated_df = dataset_from_sides(samples_arr, generation_options)
generated_df['is_valid_triangle'].value_counts(normalize=True)
```
<pre><code class="language-output">is_valid_triangle
True     0.991211
False    0.008789
Name: proportion, dtype: float64</code></pre>
### Original model conclusion {#original-model-conclusion}

Now we have a model with sufficient capability of generating valid triangles with an excellent accuracy. We can take a look at some of the valid samples we generated.


```python
fig, ax = visualize_sample_triangles(generated_df[generated_df['is_valid_triangle']])
ax.grid()
ax.set_title('Generated triangles')
plt.savefig(f"{ASSETS_FOLDER}/sample_triangle.png")
plt.close()
```

<img src="/blog/diffusion-on-the-edge-02-maximal-entropy-02-maximal-learning/assets/sample-triangle.4943f130.png" width=560 height=360>

From the above plot, we can see that our model has generated some nice triangles for us. They are very similar to the ones we saw from the original dataset.

# Maximal manifold exploration {#maximal-manifold-exploration}

Now we have the basic setup ready, we have a pretuned model $s^\text{pre}$ which is capable of generating valid triangles.
Turning our attention to the manifold $\mathcal{T}$, we have the goal of exploring some of the uncommon triangles and seeing if we can find some quite niche samples.

## Maximal exploration formalism {#maximal-exploration-formalism}

The punchline of the paper is the following: we can fine-tune an already trained model $s^\text{pre}$ to find those uncommon samples by adjusting the model. More formally, given a pre-learned model (policy) $\pi^\text{pre}$, we want to learn a new model $\pi$ such that

$$
\begin{align}
\pi &= \arg\max_{\pi} \ \mathcal{H}(p^\pi_T) \\
p^{\pi}_T &\in \mathbb{P}(\Omega_\text{pre}) \\
\end{align}
$$

is satisfied. Here $\mathcal{H}(p^\pi_T)$ denotes the entropy of the marginal density $p^\pi_T$, and the constraint ensures that $p^\pi_T$ lies within the support $\Omega_\text{pre}$ of the original pre-trained model’s distribution.

In essence, this captures the idea of learning a distribution that maximizes entropy—spreading probability mass as uniformly as possible—while remaining within the “valid” region $\Omega_\text{pre}$ defined by the pre-trained model.

## The first variation and surprise {#the-first-variation-and-surprise}

Optimizing entropy directly is difficult since it is a nonlinear functional of the distribution.
The key insight of the paper is to look at the first variation of entropy, which is linear and therefore tractable. The entropy variation evaluated at the pre-trained distribution is

$$\delta \mathcal{H}(p^\text{pre}_T)(x) = -\log p^\text{pre}_T(x)$$

which can be interpreted as a surprise signal. 
The variation makes it so samples that were unlikely under the original model are weighted highly. Thus, the fine-tuning problem can be rewritten as

$$
\pi^\ast = \arg\max_{\pi} \ \mathbb{E}_{x \sim \pi}\big[ -\log p^\text{pre}_T(x) \big] \ - \ \alpha \, D_{\text{KL}}\!\left(p^\pi_T \;\|\; p^\text{pre}_T \right),
$$

where the Kullback–Leibler regularization, $D_{\text{KL}}$, ensures that the fine-tuned policy does not drift too far and remains supported on $\Omega_\text{pre}$.

### Optional background: functionals and variations

In general, a variation is a concept from (surprise, surprise) *calculus of variations*. It is a generalization of the derivative, but instead of taking derivatives of functions, we take derivatives of functionals. A functional is a map from a space of functions into the real numbers, $\mathcal{F}: \mathcal{X} \to \mathbb{R}$ where $f \in \mathcal{X} \mapsto \mathcal{F}(f) \in \mathbb{R}$.

For example, evaluating a definite integral of an integrable function $f$ is a functional, $\mathcal{F}[f] = \int_a^b f(x)dx$. The first variation of $\mathcal{F}$ at $f$ in the direction of a perturbation $\eta(x)$ is defined as

$$
\delta \mathcal{F}[f; \eta]
= \left.\frac{d}{d\epsilon} \, \mathcal{F}[f + \epsilon \eta] \right|_{\epsilon = 0}.
$$

Here $\epsilon$ is a small parameter which acts as the strength of the disturbance. Like with directional derivatives in multivariate calculus, this is exactly the infinite-dimensional analogue.


## The crucial realization: no density estimation needed

Naively, the process of entropy maximization would require estimating $p^\text{pre}_T(x)$, which is infeasible in high dimensions. The breakthrough of the paper is recognizing that the gradient of the entropy variation is exactly equal to the negative score

$$
\nabla_x \delta \mathcal{H}(p^\pi_T) \;=\; - \nabla_x \log p^\pi_T(x) \;\approx\; - s^\pi(x, T),
$$

where $s^\pi$ is the score function. For more details, please refer to the original paper _paper name here_.

This means that instead of explicitly computing densities, we can use the score network itself to provide the update direction for entropy maximization. Fine-tuning reduces to an iterative process where the pre-trained score model supplies the gradient information needed to explore low-density regions.

## Implementation of manifold exploration in practice {#implementation-of-manifold-exploration-in-practice}

The practical implementation of the fine tuning approach is done by having a loss function, with $2$ different components
- close distance $(\Delta s_k(x, t))^2$
- tuning to direction of negative score $\left<\Delta s_k(x, T), -s_{k - 1}(x, T)\right>$

where $\Delta s_k(x, t) = s_{k}(x, t) - s_{k - 1}(x, t)$.

Adding these different loss components we obtain the total loss which is then minimized in a "student & teacher"-style mirror gradient-descent algorithm. The previous outer iteration model $s_{k - 1}$ is the teacher and the current model $s_k$ is the student. The loss function which we optimize is something like this

$$
\mathcal{L} = (\Delta s_k(x, t))^2 - \left<\Delta s_k(x, T), -s_{k - 1}(x, T)\right>
$$

which balances the distance from the original model and the tuning direction at near the end of the sample generation process. In practice, we optimize this loss using a batched gradient descent and the direction term is taken over samples where $t > (T - \epsilon)$ for some appropriate $\epsilon > 0$.



```python
teacher_tuning, student_tuning = build_teacher_student(pretrained_score_model=score_model, d = 4, hidden_size=128)
```


```python
INNER_EPOCHS = 10
OUTER_EPOCHS = 5

beta = 1.0
lambda_prox = 0.1

BATCH_SIZE = 32
DATASET_SIZE = 2 ** 10

def is_valid_triangle_torch(x_prescaled: torch.Tensor):
    x = (x_prescaled + 1) / 2
    a, b, c = torch.sort(x, dim=1).values.unbind(dim=1)
    return (a + b > c) & (a + c > b) & (b + c > a)

optimizer = torch.optim.AdamW(student_tuning.delta.parameters(), lr = 0.0001)

# Tuning loop
for epoch in range(OUTER_EPOCHS):
    # Generate new samples using score model
    x_T = torch.randn(size=(DATASET_SIZE, 4))
    with torch.no_grad():
        samples = reverse_pc_sampler_torch(
            x_T=x_T,
            process=ou_process,
            score_model=student_tuning,
            grid=grid,
            snr=0.15,
        )

        inverse_normalized = stds * samples + means # Inverse the normalization
    
    raw_lengths = inverse_normalized[:, :3] * inverse_normalized[:, [3]]
    # Check valid triangles
    validity_result = is_valid_triangle_torch(x_prescaled=raw_lengths)
    valid_samples = samples[validity_result]

    generated_dataset = OUDiffusionDatasetVectorized(
        process=ou_process,
        x0_pool=valid_samples.detach(),
        T_max=grid.t1,
        batch_size=BATCH_SIZE,
        batches_per_epoch=max(1, valid_samples.shape[0] // BATCH_SIZE),
    )
    
    generated_dataloader = torch.utils.data.DataLoader(generated_dataset, batch_size=None)
    teacher_tuning.load_state_dict(student_tuning.state_dict())
    teacher_tuning.eval()
    student_tuning.train()

    # Inner loop
    losses = []
    for inner_epoch in range(INNER_EPOCHS):
        for batch in generated_dataloader:
            optimizer.zero_grad()

            # Get training elements
            t = batch.get('t')
            x = batch.get('xt')
            score = batch.get('score')

            # Obtain a weight for the smallest t (corresponding to close the original data distribution)
            w_T = (t <= 0.1 * grid.t1).float()

            # Forward pass to tuning models
            student_prediction = student_tuning(x, t)
            with torch.no_grad():
                teacher_prediction = teacher_tuning(x, t)

            delta = student_prediction - teacher_prediction
            delta_loss = lambda_prox * torch.sum(delta ** 2, dim = 1).mean()

            # Direction loss
            dir_term = (-teacher_prediction * delta).sum(dim = 1)
            direction_loss = (beta * dir_term * w_T).mean()
            loss = delta_loss - direction_loss
            loss.backward()
            optimizer.step()

            losses.append([x_.detach().cpu().numpy() for x_ in [loss, delta_loss, direction_loss]])

    losses = np.asarray(losses)
```

## Fine-tuned model sampling {#fine-tuned-model-sampling}

Now we have a fine-tuned model $\pi^*$. Utilizing it, we can sample now triangles which are underreprensented or even not found at all in the original dataset.

We begin by sampling from the tuned dataset and examining how many of our new samples are valid.


```python
samples_tuned = reverse_pc_sampler_torch(
    x_T=x_T, # Note that we use the same input sample as the original model
    process=ou_process,
    score_model=student_tuning,
    grid=grid,
    snr=0.15,
) 
samples_tuned = samples_tuned * stds + means
samples_arr_tuned = samples_tuned.detach().cpu().numpy()
```


```python
student_generated = dataset_from_sides(samples_arr_tuned[:, :3] * samples_arr_tuned[:, [3]], generation_options)
student_generated['is_valid_triangle'].value_counts(normalize=True)
```
<pre><code class="language-output">is_valid_triangle
True     0.913086
False    0.086914
Name: proportion, dtype: float64</code></pre>
We have generated again some invalid samples. However, as we will soon see, this tradeoff is quite nice since we obtain some uncommon, but valid samples.

### Visualization {#visualization}

We have a nice proportion of valid triangles, we can visualize them in a similar way.
By creating a mapping from the side lengths $(a, b, c)$ into a shape $(u, v, 1)$ by a simple transformation, $u = a/c, v = b/c$, we can visualize the space of triangles as a submanifold of $\mathbb{R}^2$.

In this scaling, the feasible triangles are those for which $u + v \leq 1$ and $u \leq v$, these originate from the triangle inequality and the ordering which we assume on the samples generated $a \leq b \leq c$.


```python
sns.scatterplot(generated_df, x = 'a_over_c', y = 'b_over_c', alpha=0.7, marker='x').set(
    xlim=(0, 1.2), ylim=(0, 1.2), alpha = 0.3, xlabel="$a/c$", ylabel="$b/c$", title='Original vs fine-tuned model'
)
sns.scatterplot(student_generated, x = 'a_over_c', y = 'b_over_c', alpha=0.7, marker='x').set(xlim=(0, 1.2), ylim=(0, 1.20), alpha = 0.3)
plt.grid()
sns.lineplot(x = np.linspace(0, 1.2), y = np.linspace(0, 1.2), color='black', linestyle='--')
sns.lineplot(x = np.linspace(0, 1.2), y = 1 - np.linspace(0, 1.2), color='black', linestyle='--')
plt.savefig(f"{ASSETS_FOLDER}/comparison_scatterplot.png")
plt.close()
```

<img src="/blog/diffusion-on-the-edge-02-maximal-entropy-02-maximal-learning/assets/comparison-scatterplot.ac3b52ea.png" width=600 height=400>

The orange dataset is the fine-tuned model, which can be observed to be more spread out and cover more of the more exotic points when compared to the samples generated from the original diffusion model. Importantly though, the feasible region is still covered well by both models. Both the original model and the fine-tuned one generate some infeasible samples, however we can see that the fine-tuned model breaks the feasibility in a novel way in the top-right region of the plot.

### Qualitative values {#qualitative-values}

We can also examine the qualitative differences between the fine-tuned and original diffusion models.


```python
comparison_dfs = [student_generated, generated_df]
side_types_df = pd.concat([df['side_type'].value_counts(normalize=True) for df in comparison_dfs], axis = 1).fillna(0)
side_types_df.columns = ["student", "original"]
side_types_df
```
<pre><code class="language-output">student  original
side_type                      
scalene      0.964844  0.989258
isosceles    0.031250  0.010742
equilateral  0.003906  0.000000</code></pre>

```python
angle_types_df = pd.concat([df['angle_type'].value_counts(normalize=True) for df in comparison_dfs], axis = 1).fillna(0)
angle_types_df.columns = ["student", "original"]
angle_types_df
```
<pre><code class="language-output">student  original
angle_type                    
acute       0.719727  0.763672
obtuse      0.278320  0.233398
right       0.001953  0.002930</code></pre>
From the qualitative examinations, we can clearly see how our fine-tuned model exhibits similar behaviour as our original one in the well-represented categories. However, the fine-tuned one can generate more frequently samples from under-represented categories, for example generating a larger number of equilateral triangles.

## Density approximation {#density-approximation}


```python
from diffusion_on_the_edge.sde.torch_backend import pf_logp_from_x0
```


```python
density_prior_variance = float((ou_params.sigma ** 2) / (2 * ou_params.theta) * (1 - np.exp(-2 * ou_params.theta * grid.t1))) # T = 1.0
density_prior_variance
```
<pre><code class="language-output">0.187995151627489</code></pre>

```python
def make_uv_heatmap_with_pfode(
    score_model,
    ou_process,
    prior_variance,
    timegrid,
    u_range=(-1.5, 1.5), 
    v_range=(-1.5, 1.5),
    res=40, 
    n_probe=2,
    c=1.0,
    mask_fn=None,
):
    import numpy as np, torch
    us = np.linspace(*u_range, res)
    vs = np.linspace(*v_range, res)
    U, V = np.meshgrid(us, vs, indexing="xy")

    # Map (u,v) -> (x,y,z) with z=c
    xyz = np.stack([c*U.ravel(), c*V.ravel(), np.full(U.size, c)], axis=-1)
    perimeter = np.sum(xyz, axis = 1, keepdims=True)
    xyzp = np.hstack([xyz, perimeter])
    x0 = torch.tensor(xyzp, dtype=torch.float)

    logp0 = pf_logp_from_x0(
        x0, ou_process, score_model, prior_variance,
        grid=timegrid, n_probe=n_probe
    ).detach().cpu().numpy().reshape(res, res)

    # Optional: apply mask in UV-space
    if mask_fn is not None:
        M = mask_fn(U, V)   # boolean mask of shape (res,res)
        logp0 = np.where(M, logp0, np.nan)

    return logp0, us, vs
```


```python
plotting_c = 0.2
logp, _, _ = make_uv_heatmap_with_pfode(score_model, ou_process, density_prior_variance, timegrid=grid, c = plotting_c)
logp_student, _, _ = make_uv_heatmap_with_pfode(student_tuning, ou_process, density_prior_variance, timegrid=grid, c = plotting_c)

fig, ax = plt.subplots(1, 3, figsize=(12,6), constrained_layout=True)
original_map = sns.heatmap(
    logp,
    ax=ax[0],
    cbar=False,
    mask=np.isnan(logp),
    xticklabels=False, 
    yticklabels=False
)
ax[0].set_xlabel("u = a/c")
ax[0].set_ylabel("v = b/c")
ax[0].set_title('Original')
ax[0].invert_yaxis()

student_map = sns.heatmap(
    logp_student,
    ax=ax[1],
    cbar=False,
    mask=np.isnan(logp_student),
    xticklabels=False, 
    yticklabels=False
)
ax[1].set_xlabel("u = a/c")
ax[1].set_ylabel("v = b/c")
ax[1].invert_yaxis()
ax[1].set_title('Fine-tuned')

difference_map = logp_student - logp
sns.heatmap(
    difference_map,
    ax = ax[2],
    cbar=False,
    xticklabels=False,
    yticklabels=False
)
ax[2].invert_yaxis()
ax[2].set_xlabel("u = a/c")
ax[2].set_ylabel("v = b/c")
ax[2].set_title('Difference of density')

cbar = fig.colorbar(original_map.collections[0], ax=ax, location="right", shrink=0.8)
cbar.set_label("$\\log p$")

plt.savefig(f"{ASSETS_FOLDER}/heatmap_comparison.png")
plt.close()
```

<img src="/blog/diffusion-on-the-edge-02-maximal-entropy-02-maximal-learning/assets/heatmap-comparison.5d7fa9b0.png" width=868 height=400>

The above picture displays nicely the effect of our entropy increasing fine tuning. We can clearly see how the middle sections, which are well-covered by the original diffusion model, there is little difference between the models. However, in regions where the u and v coordinates are both high, the figures display clear differences in the densities. The fine-tuned model has a more spread out distribution, being indicative of increased entropy.

# Conclusion {#conclusion}

We have now implemented a mirror-descent style fine-tuning operation on our diffusion model. This implementation clearly has the ability to find undersampled samples and increase the resulting entropy of our
