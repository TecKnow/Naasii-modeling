# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo[recommended]==0.20.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "pytest==9.0.2",
#     "scipy==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    # Initialization code that runs before all other cells
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Naasii modeling
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The purpose of this notebook is to statistically model the dice game Naasii by Coyote and Crow games. This will be accomplished incrementally, starting from a single die, then multiple dice, then dice pools and so on. The basic game will also be modeled before advanced rules are added. Although modeling is at an early stage it seems most likely that the game itself will eventually be modeled with a Markov decision process.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The Naasii game

    **Naasii** is a push-your-luck dice game for 2-5 players that takes about an hour to play. Like the Coyote & Crow TTRPG, Naasii is based on 12-sided dice divided into two groups. Naasii uses 9 white Coyote dice and 3 black Crow dice.

    Players attempt to form sets of at least three of the same number or runs of at least three sequential numbers across multiple rolls on their turn. Each roll after the first provides additional white Coyote dice, but also adds a black Crow die that can cancel dice or even cause the player to bust, ending their turn early with no score. Players may end their turn after any roll where they can score.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Game resources

    - Official website: https://coyoteandcrow.net/board-game-resources/#Naasii
    - Rules: https://coyoteandcrow.net/wp-content/uploads/2025/12/Naasii-Rules-3.0.pdf
    - Scorecard: https://coyoteandcrow.net/wp-content/uploads/2023/10/Naasii-Scorecard.pdf
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interacting with this notebook

    ### Running this notebook

    ```bash
    uv run --with marimo[recommended] marimo run --sandbox naasii-modeling.py
    ```

    ### Editing this notebook

    ```bash
    uv run --with marimo[recommended] marimo edit --sandbox naasii-modeling.py
    ```

    ### Testing this notebook

    ```bash
    uv run --with-requirements naasii-modeling.py pytest naasii-modeling.py
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The foundation of Naasii gameplay: the d12

    The most basic element of Naasii gameplay is a single twelve-sided die, called a d12.  So how do we expect each d12 to behave?  We generally assume that a die is fair, meaning each face is equally likely to come up.  For a 12-sided die, we expect to roll each number $\frac{1}{12}$, or $8\frac{1}{3}\%$ of the time.

    For even the first, most basic steps of this exploration we'll need the results of more d12 rolls than anyone has the time to make, so we'll have to rely on the computer to simulate the dice rolls for us.  How do we convince ourselves that we can trust the simulation?  And once we are convinced that we can trust the simulated dice, how do we quanitfy that trust to communicate it to others?

    We can do this incrementally.
    1. First, simulate a large number of die rolls and use graphs and tables to decide if the results look like we would expect.
    2. Find an ideal statistical distribution that models our situation and compare our results to this ideal.
    3. Use math to put a number on our uncertainty.  What is the probability that we're wrong?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Manual inspection of d12 rolls
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
    single_die_trials = int(uniform_trials.value)
    single_die_rolls = roll_d12s(single_die_trials, num_dice=1).ravel()
    return single_die_rolls, single_die_trials


@app.cell(hide_code=True)
def _(single_die_rolls, single_die_trials):
    single_die_faces = np.arange(1, SIDES + 1)
    single_die_counts = np.bincount(single_die_rolls, minlength=SIDES + 1)[single_die_faces]
    single_die_probabilities = single_die_counts / single_die_trials
    exact_single_die_probability = 1 / SIDES
    single_die_max_deviation = np.abs(
        single_die_probabilities - exact_single_die_probability
    ).max()
    single_die_chi_square, single_die_expected_count, single_die_df = (
        chi_squared_uniformity(single_die_counts)
    )
    single_die_critical_value = chi_squared_critical_value(single_die_df)
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
def _(
    exact_single_die_probability,
    single_die_faces,
    single_die_probabilities,
):
    uniformity_rows = "\n".join(
        f"| {int(face)} | {probability:.3%} | {probability - exact_single_die_probability:+.3%} |"
        for face, probability in zip(single_die_faces, single_die_probabilities)
    )
    mo.md(
        "\n".join(
            [
                "Measured frequencies by face compared with the exact uniform model:",
                "",
                "| Face | Measured frequency | Difference from expected (1/12) |",
                "| --- | ---: | ---: |",
                uniformity_rows,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Measuring the distance to our expectations: Chi-squared

    The graph and table in the previous section show us how much more or less likely each face was in practice compared to our expectations.  We cannot just add them up because positive and negative deviations would cancel out.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To make the visual check more precise, we also compute Pearson's chi-squared
    goodness-of-fit statistic

    \[
    \chi^2 = \sum\_{i=1}^{12} \frac{(O_i - E_i)^2}{E_i},
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
    \(\chi^2\_{11}\). The \(11\) degrees of freedom come from the fact that there are
    \(12\) face counts, but once \(11\) of them are known, the last one is fixed because
    the counts must sum to the total number of rolls.

    In a full statistics course, one often reports a p-value. This notebook uses the
    equivalent critical-value view instead: for a 5% test, the cutoff is about
    \(19.675\). More precisely, this number is chosen so that

    \[
    P(\chi^2\_{11} \le 19.675) \approx 0.95,
    \]

    which means \(19.675\) is the 95th percentile of the \(\chi^2\_{11}\)
    distribution, or equivalently the point that leaves 5% of the distribution in the
    upper tail.

    Historically, one would usually look this up in a chi-squared table. In modern
    practice, one often gets it from software. For example, statistics software would
    compute the same cutoff as the inverse CDF, or quantile, of \(\chi^2\_{11}\) at
    probability \(0.95\). In this notebook we compute the cutoff directly from the
    chi-squared distribution, so the same logic can be reused for other degrees of
    freedom as well.

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
        chi_square_conclusion = "This sample would be flagged by the 5% chi-squared check."

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


@app.cell
def _():
    mo.md(r"""
    ### Interactive chi-squared explorer

    The single-d12 goodness-of-fit test fixes the degrees of freedom at \(11\). Use the
    controls below to see how the chi-squared curve and right-tail rejection region
    change for other degrees of freedom and significance levels.
    """)
    return


@app.cell
def _():
    chi_squared_df = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=11,
        show_value=True,
        label="Degrees of freedom",
    )
    chi_squared_alpha = mo.ui.dropdown(
        options={"10%": 0.10, "5%": 0.05, "1%": 0.01},
        value="5%",
        label="Significance level",
    )
    mo.vstack(
        [
            chi_squared_df,
            chi_squared_alpha,
        ]
    )
    return chi_squared_alpha, chi_squared_df


@app.cell(hide_code=True)
def _(chi_squared_alpha, chi_squared_df):
    _explorer_alpha = float(chi_squared_alpha.value)
    _explorer_df = int(chi_squared_df.value)
    _explorer_critical_value = chi_squared_critical_value(
        _explorer_df, alpha=_explorer_alpha
    )
    _explorer_x_max = max(
        float(scipy.stats.chi2.ppf(0.999, df=_explorer_df)),
        _explorer_critical_value * 1.1,
    )
    _explorer_x = np.linspace(0.0, _explorer_x_max, num=600)
    _explorer_pdf = scipy.stats.chi2.pdf(_explorer_x, df=_explorer_df)
    _explorer_pdf[~np.isfinite(_explorer_pdf)] = np.nan

    _explorer_fig, _explorer_ax = plt.subplots(figsize=(10, 4))
    _explorer_ax.plot(
        _explorer_x,
        _explorer_pdf,
        color="tab:blue",
        label=f"$\\chi^2_{{{_explorer_df}}}$ density",
    )
    _explorer_ax.axvline(
        _explorer_critical_value,
        color="tab:red",
        linestyle="--",
        label=f"Critical value = {_explorer_critical_value:.3f}",
    )
    _explorer_ax.fill_between(
        _explorer_x,
        0,
        _explorer_pdf,
        where=_explorer_x >= _explorer_critical_value,
        color="tab:red",
        alpha=0.25,
        label=f"Right-tail area = {_explorer_alpha:.0%}",
    )
    _explorer_ax.set_xlim(0, _explorer_x_max)
    _explorer_ax.set_title(
        f"Chi-squared density with {_explorer_df} degrees of freedom"
    )
    _explorer_ax.set_xlabel("Chi-squared value")
    _explorer_ax.set_ylabel("Density")
    _explorer_ax.grid(axis="y", alpha=0.2)
    _explorer_ax.legend()
    _explorer_fig.tight_layout()
    _explorer_fig
    return


@app.cell
def _(chi_squared_alpha, chi_squared_df):
    _explorer_alpha = float(chi_squared_alpha.value)
    _explorer_df = int(chi_squared_df.value)
    _explorer_critical_value = chi_squared_critical_value(
        _explorer_df, alpha=_explorer_alpha
    )
    mo.md(f"""
    With **{_explorer_df}** degrees of freedom and significance level
    **{_explorer_alpha:.0%}**, the right-tail critical value is
    **{_explorer_critical_value:.3f}**.

    The shaded region marks the rejection tail: values to the right of the cutoff
    have total probability **{_explorer_alpha:.0%}** under the selected
    chi-squared model.
    """)
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
    multi_trial_count = int(multi_trials.value)
    dice_count = int(num_dice.value)
    requested_target = int(target_sum.value)
    effective_target = int(
        np.clip(requested_target, dice_count, dice_count * SIDES)
    )
    rolls = roll_d12s(multi_trial_count, dice_count)
    totals = rolls.sum(axis=1)

    possible_totals, exact_probabilities = exact_sum_distribution(dice_count)
    simulated_counts = np.bincount(totals, minlength=possible_totals[-1] + 1)
    simulated_probabilities = simulated_counts[possible_totals] / multi_trial_count

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


@app.function
def roll_d12s(
    trials: int, num_dice: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Return a trials-by-num_dice array of simulated d12 rolls."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(1, SIDES, size=(trials, num_dice), endpoint=True)


@app.function(hide_code=True)
def chi_squared_uniformity(observed_counts: np.ndarray) -> tuple[float, float, int]:
    """Compute Pearson's chi-squared statistic for a uniform categorical distribution."""
    if observed_counts.ndim != 1:
        raise ValueError("observed_counts must be one-dimensional")

    expected_count = observed_counts.sum() / observed_counts.size
    chi_square = np.sum((observed_counts - expected_count) ** 2 / expected_count)
    degrees_freedom = observed_counts.size - 1
    return float(chi_square), float(expected_count), int(degrees_freedom)


@app.function
def chi_squared_critical_value(degrees_freedom: int, alpha: float = 0.05) -> float:
    """Return the right-tail critical value for a chi-squared test."""
    return float(scipy.stats.chi2.ppf(1 - alpha, df=degrees_freedom))


@app.function(hide_code=True)
def exact_sum_distribution(
    num_dice: int, sides: int = SIDES
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact probability distribution for the sum of fair dice."""
    counts = np.ones(sides, dtype=np.int64)
    for _ in range(1, num_dice):
        counts = np.convolve(counts, np.ones(sides, dtype=np.int64))

    totals = np.arange(num_dice, num_dice * sides + 1)
    probabilities = counts / counts.sum()
    return totals, probabilities


@app.function(hide_code=True)
def running_event_rate(
    event_hits: np.ndarray, points: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the cumulative event rate at evenly spaced checkpoints."""
    sample_sizes = np.unique(
        np.linspace(1, event_hits.size, num=min(points, event_hits.size), dtype=int)
    )
    cumulative_hits = np.cumsum(event_hits.astype(np.int64))
    return sample_sizes, cumulative_hits[sample_sizes - 1] / sample_sizes


@app.cell
def _():
    def test_roll_d12s_shape():
        rng = np.random.default_rng(7)
        rolls = roll_d12s(trials=7, num_dice=3, rng=rng)

        assert rolls.shape == (7, 3)


    def test_roll_d12s_integer_bounds():
        rng = np.random.default_rng(17)
        rolls = roll_d12s(trials=200, num_dice=4, rng=rng)

        assert np.issubdtype(rolls.dtype, np.integer)
        assert rolls.min() >= 1
        assert rolls.max() <= SIDES


    def test_roll_d12s_seed_reproducibility():
        rng_a = np.random.default_rng(23)
        rng_b = np.random.default_rng(23)

        rolls_a = roll_d12s(trials=25, num_dice=2, rng=rng_a)
        rolls_b = roll_d12s(trials=25, num_dice=2, rng=rng_b)

        assert np.array_equal(rolls_a, rolls_b)


    def test_roll_d12s_default_rng_created_per_call():
        from unittest.mock import patch

        rng_a = np.random.default_rng(31)
        rng_b = np.random.default_rng(31)

        with patch.object(np.random, "default_rng", side_effect=[rng_a, rng_b]) as default_rng:
            rolls_a = roll_d12s(trials=25, num_dice=2)
            rolls_b = roll_d12s(trials=25, num_dice=2)

        assert default_rng.call_count == 2
        assert np.array_equal(rolls_a, rolls_b)


    def test_chi_squared_uniformity_uniform_counts():
        chi_square, expected_count, degrees_freedom = chi_squared_uniformity(
            np.array([5, 5, 5, 5])
        )

        assert chi_square == 0.0
        assert expected_count == 5.0
        assert degrees_freedom == 3


    def test_chi_squared_uniformity_known_counts():
        chi_square, expected_count, degrees_freedom = chi_squared_uniformity(
            np.array([8, 2, 5, 5])
        )

        assert np.isclose(chi_square, 3.6)
        assert expected_count == 5.0
        assert degrees_freedom == 3


    def test_chi_squared_uniformity_requires_one_dimension():
        try:
            chi_squared_uniformity(np.ones((2, 2), dtype=np.int64))
        except ValueError as exc:
            assert str(exc) == "observed_counts must be one-dimensional"
        else:
            assert False, "Expected ValueError for non-1D observed_counts"


    def test_chi_squared_critical_value_matches_known_cutoff():
        critical_value = chi_squared_critical_value(11, alpha=0.05)

        assert np.isclose(critical_value, 19.675, atol=0.001)


    def test_chi_squared_critical_value_changes_with_alpha():
        strict_cutoff = chi_squared_critical_value(11, alpha=0.01)
        loose_cutoff = chi_squared_critical_value(11, alpha=0.10)

        assert strict_cutoff > loose_cutoff


    def test_exact_sum_distribution_single_die_is_uniform():
        totals, probabilities = exact_sum_distribution(num_dice=1)

        assert np.array_equal(totals, np.arange(1, SIDES + 1))
        assert np.allclose(probabilities, np.full(SIDES, 1 / SIDES))


    def test_exact_sum_distribution_two_dice_known_probabilities():
        totals, probabilities = exact_sum_distribution(num_dice=2)

        assert totals[0] == 2
        assert totals[-1] == 2 * SIDES
        assert np.isclose(probabilities[0], 1 / SIDES**2)
        assert np.isclose(probabilities[-1], 1 / SIDES**2)
        assert np.isclose(probabilities[SIDES - 1], 1 / SIDES)


    def test_exact_sum_distribution_probabilities_sum_to_one():
        _, probabilities = exact_sum_distribution(num_dice=4)

        assert np.isclose(probabilities.sum(), 1.0)


    def test_running_event_rate_uses_all_points_when_input_is_short():
        sample_sizes, running_rates = running_event_rate(
            np.array([True, False, True, True]), points=30
        )

        assert np.array_equal(sample_sizes, np.array([1, 2, 3, 4]))
        assert np.allclose(running_rates, np.array([1.0, 0.5, 2 / 3, 0.75]))


    def test_running_event_rate_respects_requested_checkpoints():
        sample_sizes, running_rates = running_event_rate(
            np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1]), points=3
        )

        assert np.array_equal(sample_sizes, np.array([1, 5, 10]))
        assert np.allclose(running_rates, np.array([1.0, 0.6, 0.5]))

    return


if __name__ == "__main__":
    app.run()
