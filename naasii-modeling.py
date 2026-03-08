# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.20.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    SIDES = 12
    TRIAL_STEPS = (
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
    )


@app.function(hide_code=True)
def roll_d12s(rng: np.random.Generator, trials: int, num_dice: int) -> np.ndarray:
    """Return a trials-by-num_dice array of simulated d12 rolls."""
    return rng.integers(1, SIDES, size=(trials, num_dice), endpoint=True)


@app.function(hide_code=True)
def exact_sum_distribution(num_dice: int, sides: int = SIDES) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact probability distribution for the sum of fair dice."""
    counts = np.ones(sides, dtype=np.int64)
    for _ in range(1, num_dice):
        counts = np.convolve(counts, np.ones(sides, dtype=np.int64))

    totals = np.arange(num_dice, num_dice * sides + 1)
    probabilities = counts / counts.sum()
    return totals, probabilities


@app.function(hide_code=True)
def running_event_rate(event_hits: np.ndarray, points: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Sample the cumulative event rate at evenly spaced checkpoints."""
    sample_sizes = np.unique(
        np.linspace(1, event_hits.size, num=min(points, event_hits.size), dtype=int)
    )
    cumulative_hits = np.cumsum(event_hits.astype(np.int64))
    return sample_sizes, cumulative_hits[sample_sizes - 1] / sample_sizes


@app.function(hide_code=True)
def chi_squared_uniformity(observed_counts: np.ndarray) -> tuple[float, float, int]:
    """Compute Pearson's chi-squared statistic for a uniform categorical distribution."""
    if observed_counts.ndim != 1:
        raise ValueError("observed_counts must be one-dimensional")

    expected_count = observed_counts.sum() / observed_counts.size
    chi_square = np.sum((observed_counts - expected_count) ** 2 / expected_count)
    degrees_freedom = observed_counts.size - 1
    return float(chi_square), float(expected_count), int(degrees_freedom)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Naasii modeling

    This notebook starts with the statistics basics we will need later for Naasii:
    exact probability distributions, simulation with NumPy, and plots that react to
    a few Marimo controls.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why start here?

    Naasii uses twelve-sided dice, so the examples below use fair d12s. The goal of
    this first pass is not to model the game rules yet. It is to verify the notebook
    workflow and build intuition for:

    - exact probability vs simulated probability
    - how sample size affects stability
    - how to define events that we will later reuse for Naasii decisions

    Later increments can replace these generic events with Naasii-specific ones such
    as making a set, making a run, or surviving Crow dice.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Uniformity check for one d12

    Before summing multiple dice, verify the base ingredient: a fair twelve-sided die.
    We simulate many single-die rolls and compare the observed face frequencies to the
    exact probability of **1/12** for each face.
    """)
    return


@app.cell(hide_code=True)
def _():
    uniform_trials = mo.ui.slider(
        steps=TRIAL_STEPS,
        value=20_000,
        label="Single-die simulation trials",
    )
    mo.vstack(
        [
            mo.md(
                "Adjust the trial count for the single-die simulation, then compare the observed face frequencies to the exact uniform model. The slider uses log-spaced values so you can jump quickly between very small and very large samples."
            ),
            uniform_trials,
        ]
    )
    return (uniform_trials,)


@app.cell(hide_code=True)
def _(uniform_trials):
    uniform_rng = np.random.default_rng()
    single_die_rolls = roll_d12s(uniform_rng, uniform_trials.value, num_dice=1).ravel()
    single_die_faces = np.arange(1, SIDES + 1)
    single_die_counts = np.bincount(single_die_rolls, minlength=SIDES + 1)[single_die_faces]
    single_die_probabilities = single_die_counts / uniform_trials.value
    exact_single_die_probability = 1 / SIDES
    single_die_max_deviation = np.abs(
        single_die_probabilities - exact_single_die_probability
    ).max()
    single_die_chi_square, single_die_expected_count, single_die_df = (
        chi_squared_uniformity(single_die_counts)
    )
    single_die_critical_value = 19.675
    single_die_uniformity_passes = single_die_chi_square <= single_die_critical_value
    return (
        exact_single_die_probability,
        single_die_chi_square,
        single_die_critical_value,
        single_die_df,
        single_die_expected_count,
        single_die_faces,
        single_die_max_deviation,
        single_die_probabilities,
        single_die_uniformity_passes,
    )


@app.cell(hide_code=True)
def _(
    exact_single_die_probability,
    single_die_faces,
    single_die_probabilities,
):
    uniform_fig, uniform_ax = plt.subplots(figsize=(10, 4))
    uniform_ax.bar(
        single_die_faces,
        single_die_probabilities,
        width=0.7,
        label="Simulated frequency",
    )
    uniform_ax.axhline(
        exact_single_die_probability,
        color="black",
        linestyle="--",
        label="Exact probability (1/12)",
    )
    uniform_ax.set_title("Single-d12 frequencies approach a uniform distribution")
    uniform_ax.set_xlabel("Face value")
    uniform_ax.set_ylabel("Probability")
    uniform_ax.set_xticks(single_die_faces)
    uniform_ax.legend()
    uniform_ax.grid(axis="y", alpha=0.2)
    uniform_fig.tight_layout()
    uniform_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To make the visual check more precise, we also compute Pearson's chi-squared
    goodness-of-fit statistic

    \[
    \chi^2 = \sum_{i=1}^{12} \frac{(O_i - E_i)^2}{E_i},
    \]

    where \(O_i\) is the observed count for face \(i\), and \(E_i = n/12\) is the
    expected count after \(n\) rolls if the die is fair.

    This statistic asks a simple question: if the die really is uniform, how far away
    are the observed counts from the counts we would expect just from random sampling?

    Each term in the sum measures the discrepancy for one face:

    - \(O_i - E_i\) is the raw difference between observed and expected counts.
    - Squaring that difference makes large mismatches matter more and prevents positive
      and negative deviations from cancelling out.
    - Dividing by \(E_i\) scales the discrepancy relative to the amount of variation we
      would naturally expect at that count level.

    So a small value of \(\chi^2\) means the observed frequencies are close to what a
    fair die would plausibly produce, while a large value means the discrepancies are
    larger than we would usually expect from random variation alone.

    Under the fair-die model, this statistic is approximately distributed as
    \(\chi^2_{11}\). The \(11\) degrees of freedom come from the fact that there are
    \(12\) face counts, but once \(11\) of them are known, the last one is fixed because
    the counts must sum to the total number of rolls.

    In a full statistics course, one often reports a p-value. This notebook uses the
    equivalent critical-value view instead: for a 5% test, the cutoff is about
    \(19.675\). More precisely, this number is chosen so that

    \[
    P(\chi^2_{11} \le 19.675) \approx 0.95,
    \]

    which means \(19.675\) is the 95th percentile of the \(\chi^2_{11}\)
    distribution, or equivalently the point that leaves 5% of the distribution in the
    upper tail.

    Historically, one would usually look this up in a chi-squared table. In modern
    practice, one often gets it from software. For example, statistics software would
    compute the same cutoff as the inverse CDF, or quantile, of \(\chi^2_{11}\) at
    probability \(0.95\). In this notebook the value is hard-coded because the number of
    faces is fixed, so the degrees of freedom are fixed at \(11\).

    - If \(\chi^2 \le 19.675\), the sample is considered consistent with a fair d12 at
      the 5% level.
    - If \(\chi^2 > 19.675\), the sample is unusual enough under the fair-die model
      that we would flag it as evidence against uniformity.

    This is still a probabilistic check, not a proof. Even a fair die will exceed the
    5% cutoff about 1 time in 20 just by random variation. Failing the test once does
    not prove bias, and passing it does not prove fairness. It only tells us whether
    the observed pattern is surprising under the uniform model.
    """)
    return


@app.cell(hide_code=True)
def _(
    exact_single_die_probability,
    single_die_chi_square,
    single_die_critical_value,
    single_die_df,
    single_die_expected_count,
    single_die_max_deviation,
    single_die_uniformity_passes,
    uniform_trials,
):
    chi_square_interpretation = "does not exceed"
    chi_square_conclusion = "This sample is consistent with a fair d12 at the 5% level."
    if not single_die_uniformity_passes:
        chi_square_interpretation = "exceeds"
        chi_square_conclusion = (
            "This sample would be flagged by the 5% chi-squared check."
        )

    mo.md(
        f"""
        With **{uniform_trials.value:,} simulated single-die rolls**, each face should land near
        **{exact_single_die_probability:.3%}**. The largest absolute deviation from the
        exact probability is **{single_die_max_deviation:.3%}**.

        The expected count per face is **{single_die_expected_count:.1f}**. The observed
        chi-squared statistic is **{single_die_chi_square:.3f}** with
        **{single_die_df}** degrees of freedom, which **{chi_square_interpretation}**
        the 5% critical value **{single_die_critical_value:.3f}**.

        {chi_square_conclusion}

        Interpreted informally: this tells us whether the overall pattern of face counts
        looks like ordinary sampling noise around a uniform distribution, not just
        whether one face happened to be a little high or low.

        As the number of trials grows, that deviation should usually shrink and the bars
        should flatten toward a uniform distribution.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exact distribution and Monte Carlo estimate

    The single-die section established the model for one fair d12: each face has
    probability \(1/12\). The next step is to treat several dice as independent copies
    of that same random variable and study the **sum** of their outcomes.

    That changes the question in an important way:

    - For one die, we care about the frequency of each face.
    - For multiple dice, we care about the distribution of the total.

    The key new idea is **convolution**. In this setting, convolution is the rule for
    combining probability distributions when we add independent random variables.

    If \(X\) and \(Y\) are two independent dice, then for any total \(t\),

    \[
    P(X + Y = t) = \sum_k P(X = k)P(Y = t-k).
    \]

    In words: to find the probability of a total like \(t=7\), we add up the
    probabilities of all pairs of face values that produce that total:
    \((1,6), (2,5), \dots, (6,1)\).

    That is exactly what convolution does. It takes the one-die distribution and
    combines it with itself to produce the two-dice distribution. Repeating that process
    gives the exact distribution for three dice, four dice, and so on.

    For the selected number of d12s, we compute the exact distribution of the total by
    convolving the one-die distribution with itself. Then we simulate many rolls with
    NumPy, sum across each row of dice, and check how closely the simulation matches the
    exact result.
    """)
    return


@app.cell(hide_code=True)
def _():
    multi_trials = mo.ui.slider(
        steps=TRIAL_STEPS,
        value=20_000,
        label="Multiple-dice simulation trials",
    )
    num_dice = mo.ui.slider(
        start=1,
        stop=4,
        step=1,
        value=3,
        label="Number of d12s",
    )
    target_sum = mo.ui.slider(
        start=1,
        stop=SIDES * 4,
        step=1,
        value=21,
        label="Target total",
    )
    mo.vstack(
        [
            mo.md(
                "Adjust the controls for the multiple-dice model, then compare the exact total distribution to a Monte Carlo simulation. The trial slider uses log-spaced values so you can move quickly from noisy small samples to stable large ones."
            ),
            multi_trials,
            num_dice,
            target_sum,
        ]
    )
    return multi_trials, num_dice, target_sum


@app.cell(hide_code=True)
def _(multi_trials, num_dice, target_sum):
    sum_rng = np.random.default_rng()
    effective_target = int(np.clip(target_sum.value, num_dice.value, num_dice.value * SIDES))
    rolls = roll_d12s(sum_rng, multi_trials.value, num_dice.value)
    totals = rolls.sum(axis=1)

    possible_totals, exact_probabilities = exact_sum_distribution(num_dice.value)
    simulated_counts = np.bincount(totals, minlength=possible_totals[-1] + 1)
    simulated_probabilities = simulated_counts[possible_totals] / multi_trials.value

    event_hits = totals >= effective_target
    event_sample_sizes, event_running_rates = running_event_rate(event_hits)
    exact_event_probability = exact_probabilities[possible_totals >= effective_target].sum()
    simulated_event_probability = event_hits.mean()

    exact_expected_total = np.dot(possible_totals, exact_probabilities)
    simulated_expected_total = totals.mean()
    return (
        effective_target,
        event_running_rates,
        event_sample_sizes,
        exact_event_probability,
        exact_expected_total,
        exact_probabilities,
        possible_totals,
        simulated_event_probability,
        simulated_expected_total,
        simulated_probabilities,
        totals,
    )


@app.cell(hide_code=True)
def _(exact_probabilities, possible_totals, simulated_probabilities):
    dist_fig, dist_ax = plt.subplots(figsize=(10, 4.5))
    dist_ax.bar(
        possible_totals - 0.2,
        exact_probabilities,
        width=0.4,
        label="Exact probability",
    )
    dist_ax.bar(
        possible_totals + 0.2,
        simulated_probabilities,
        width=0.4,
        alpha=0.75,
        label="Simulated probability",
    )
    dist_ax.set_title("Distribution of the total")
    dist_ax.set_xlabel("Total rolled")
    dist_ax.set_ylabel("Probability")
    dist_ax.legend()
    dist_ax.grid(axis="y", alpha=0.2)
    dist_fig.tight_layout()
    dist_fig
    return


@app.cell(hide_code=True)
def _(
    effective_target,
    exact_event_probability,
    exact_expected_total,
    multi_trials,
    num_dice,
    simulated_event_probability,
    simulated_expected_total,
    target_sum,
    totals,
):
    clipped_note = ""
    if target_sum.value != effective_target:
        clipped_note = (
            f"Target total was clipped to **{effective_target}** because "
            f"{num_dice.value} d12s cannot sum to **{target_sum.value}**."
        )

    mo.md(
        f"""
        ## Read the output

        With **{num_dice.value} d12s** and **{multi_trials.value:,} simulated rolls**:

        - Exact expected total: **{exact_expected_total:.3f}**
        - Simulated expected total: **{simulated_expected_total:.3f}**
        - Exact probability of rolling at least **{effective_target}**: **{exact_event_probability:.3%}**
        - Simulated probability of rolling at least **{effective_target}**: **{simulated_event_probability:.3%}**
        - Smallest simulated total: **{totals.min()}**
        - Largest simulated total: **{totals.max()}**

        {clipped_note}

        The gap between exact and simulated values should usually shrink as you increase
        the number of trials.
        """
    )
    return


@app.cell(hide_code=True)
def _(
    effective_target,
    event_running_rates,
    event_sample_sizes,
    exact_event_probability,
):
    conv_fig, conv_ax = plt.subplots(figsize=(10, 4))
    conv_ax.plot(
        event_sample_sizes,
        event_running_rates,
        linewidth=2,
        label="Running simulated estimate",
    )
    conv_ax.axhline(
        exact_event_probability,
        color="black",
        linestyle="--",
        label=f"Exact P(total >= {effective_target})",
    )
    conv_ax.set_title("Simulation converges toward the exact probability")
    conv_ax.set_xlabel("Number of simulated rolls used")
    conv_ax.set_ylabel("Probability")
    conv_ax.legend()
    conv_ax.grid(alpha=0.2)
    conv_fig.tight_layout()
    conv_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Connection to Naasii

    This notebook now demonstrates the three ingredients we will need for the game model:

    - represent die outcomes as arrays
    - define an event and estimate its probability by simulation
    - compare a simulation to an exact result whenever an exact result is still tractable

    A natural next increment is to replace the generic `total >= target` event with a
    Naasii event on the opening roll, such as:

    - probability of at least one scoreable set
    - probability of at least one scoreable run
    - probability of any scoreable set or run
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Game resources

    - Official website: https://coyoteandcrow.net/board-game-resources/#Naasii
    - Rules: https://coyoteandcrow.net/wp-content/uploads/2025/12/Naasii-Rules-3.0.pdf
    - Scorecard: https://coyoteandcrow.net/wp-content/uploads/2023/10/Naasii-Scorecard.pdf

    ## To edit

    ```bash
    uv run --with marimo[recommended] marimo edit --sandbox naasii-modeling.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
