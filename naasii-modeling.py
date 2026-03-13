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
    import math
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
    This notebook uses probability, simulation, and Python to study the dice game Naasii by Coyote and Crow games. We begin with one fair d12, move to several fair d12s, and then build toward the scoreable patterns and special rules that matter in the game itself.
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


@app.cell
def _():
    mo.md(r"""
    ## How should a real d12 behave?

    The most basic element of Naasii gameplay is a single twelve-sided die, called a d12. A natural first question is: if a d12 is fair, what should its long-run behavior look like? We generally assume that a fair die gives each face the same chance to appear. For a 12-sided die, that means each number should come up with probability $\frac{1}{12}$, or $8\frac{1}{3}\%$.

    To study that behavior, we want far more rolls than anyone would want to do by hand. So we let the computer imitate many rolls for us. This is called a **simulation**: the computer uses a random number generator to stand in for repeated dice rolls. Once we have that simulation, two more questions immediately follow. How do we decide whether the simulated d12 behaves like a believable fair die? And if it does, how do we measure how close it is to the ideal uniform model?

    We can do this incrementally.
    1. First, simulate a large number of die rolls and use graphs and tables to see whether the results look like a fair die.
    2. Then compare those results with the exact uniform model.
    3. Finally, use probability to measure whether the overall mismatch is small enough to be believable.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Do the simulated rolls look like a fair d12?
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
                "Adjust the number of simulated rolls, then compare the observed face frequencies with the exact uniform model. The slider uses log-spaced values so you can jump quickly between very small and very large samples."
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


@app.cell
def _():
    mo.md(r"""
    ### How can we measure the overall mismatch?

    The graph and table in the previous section show how much more or less often each face appeared than we expected. But those positive and negative differences can cancel each other out, so we need a better way to measure the total mismatch.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    One standard way to measure that mismatch is **Pearson's chi-squared goodness-of-fit statistic**:

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

    One common way to report this calculation is with a p-value. In this notebook we
    use the equivalent critical-value view instead. For a 5% test, the cutoff is about
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
    ### What does the chi-squared distribution measure?

    The single-d12 goodness-of-fit test fixes the degrees of freedom at \(11\). The
    explorer below is a deeper look at the distribution behind that test. Use it to see
    how the chi-squared curve and right-tail rejection region change for other degrees
    of freedom and significance levels.
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
    _explorer_ax.set_title(f"Chi-squared density with {_explorer_df} degrees of freedom")
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
    For the main story of the notebook, the important conclusion is simpler: a single
    simulated d12 can be checked against the uniform model in a principled way. But
    Naasii is a dice-pool game, so one die is only the beginning. The next question is
    what happens when several fair d12s are rolled together.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What totals should we expect from several fair d12s?

    The single-die section established the model for one fair d12: each face has
    probability \(1/12\). Now we treat several dice as independent copies of that same
    random variable and study the **sum** of their outcomes.

    This changes the question in an important way:

    - For one die, we care about the frequency of each face.
    - For multiple dice, we care about the distribution of the total.

    There are two complementary ways to study those totals. One way is exact: compute
    the probability of every possible total. The other way is approximate: let the
    computer generate many random samples and estimate the probabilities from those
    samples.

    The exact calculation uses **convolution**. In this setting, convolution is the
    rule for combining probability distributions when we add independent random
    variables.

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

    The approximate method is often called a **Monte Carlo estimate**: we use repeated
    random sampling to estimate probabilities numerically.

    For the selected number of d12s, we compute the exact distribution of the total by
    convolving the one-die distribution with itself. Then we generate many simulated
    rolls with NumPy, sum across each row of dice, and check how closely the simulation
    matches the exact result.
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
                "Adjust the controls for several fair d12s, then compare the exact total distribution with the estimate built from repeated random sampling. The trial slider uses log-spaced values so you can move quickly from noisy small samples to stable large ones."
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
    effective_target = int(np.clip(requested_target, dice_count, dice_count * SIDES))
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
        ### What do these results tell us?

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

        Totals are a useful first model for several dice because we can compute them
        exactly and estimate them by simulation. But Naasii scoring is not based only on
        large totals. Players score by forming patterns such as sets and runs, so the
        next question is how often those patterns appear.
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
    ## When do several d12s make a scoreable set?

    In Naasii, a set becomes scoreable at three of a kind. Larger sets are better
    because a set is worth its size: a 3-set is worth 3 points, a 4-set is worth 4
    points, and so on.

    That makes sets a natural place to introduce **counting** in probability. We first
    ask about one chosen face, such as "How likely is it that 7 appears at least three
    times?" Then we generalize to the harder question "How likely is it that some face
    appears at least three times?"
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What does \(\binom{n}{k}\) mean?

    We read \(\binom{n}{k}\) as **"n choose k."** It counts how many ways we can choose
    which \(k\) of the \(n\) dice show the chosen face, without caring about the order
    in which we list those positions.
    """)
    return


@app.cell(hide_code=True)
def _():
    combination_example_num_dice = mo.ui.slider(
        start=3,
        stop=6,
        step=1,
        value=4,
        label="Example number of dice n",
    )
    return (combination_example_num_dice,)


@app.cell(hide_code=True)
def _(combination_example_num_dice):
    _example_num_dice_value = int(combination_example_num_dice.value)
    combination_example_match_count = mo.ui.slider(
        start=0,
        stop=_example_num_dice_value,
        step=1,
        value=min(2, _example_num_dice_value),
        label="Example chosen-face count k",
    )
    return (combination_example_match_count,)


@app.cell(hide_code=True)
def _(combination_example_match_count, combination_example_num_dice):
    mo.vstack(
        [
            mo.md(
                "Use a smaller example first. Each row below names a different set of die positions that could show the chosen face."
            ),
            combination_example_num_dice,
            combination_example_match_count,
        ]
    )
    return


@app.cell(hide_code=True)
def _(combination_example_match_count, combination_example_num_dice):
    _example_num_dice_value = int(combination_example_num_dice.value)
    _example_match_count_value = int(combination_example_match_count.value)
    _position_choices = enumerate_position_choices(
        _example_num_dice_value, _example_match_count_value
    )
    _remaining_dice = _example_num_dice_value - _example_match_count_value
    _completions_per_choice = (SIDES - 1) ** _remaining_dice
    _shortcut_value = math.factorial(_example_num_dice_value) // (
        math.factorial(_example_match_count_value) * math.factorial(_remaining_dice)
    )


    def _format_positions(position_choice: tuple[int, ...]) -> str:
        if not position_choice:
            return "none"
        return ", ".join(str(position) for position in position_choice)


    _position_choice_rows = "\n".join(
        f"| {index} | {_format_positions(position_choice)} |"
        for index, position_choice in enumerate(_position_choices, start=1)
    )

    if _remaining_dice == 0:
        _completion_note = (
            "Here there are no remaining dice, so each row already determines a full "
            "favorable outcome."
        )
    else:
        _completion_note = (
            f"Once one row is fixed, the remaining **{_remaining_dice}** dice can each "
            f"be any of the other 11 faces, so each row can be completed in "
            f"$11^{{{_remaining_dice}}} = {_completions_per_choice:,}$ ways."
        )

    mo.md(
        "\n".join(
            [
                f"For this example, $\\binom{{{_example_num_dice_value}}}{{{_example_match_count_value}}}$ means the number of ways to choose which **{_example_match_count_value}** of the **{_example_num_dice_value}** die positions show the chosen face. We care about **which positions**, not the order in which we name them.",
                "",
                "| Choice | Die positions showing the chosen face |",
                "| ---: | --- |",
                _position_choice_rows,
                "",
                f"There are **{len(_position_choices)}** position choices, so $\\binom{{{_example_num_dice_value}}}{{{_example_match_count_value}}} = {len(_position_choices)}$.",
                "",
                _completion_note,
                "",
                f"A compact shortcut for the same count is $\\binom{{{_example_num_dice_value}}}{{{_example_match_count_value}}} = \\frac{{{_example_num_dice_value}!}}{{{_example_match_count_value}!({_remaining_dice})!}} = {_shortcut_value}$. This shortcut counts the same position choices more efficiently; it is not a new probability rule.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What does counting mean in probability?

    When all \(12^n\) ordered outcomes of \(n\) fair d12s are equally likely, an event
    probability can be written as

    \[
    P(E) = \frac{\#\text{ favorable outcomes}}{\#\text{ total outcomes}}.
    \]

    For one chosen face and a fixed set size \(k\), the exact-\(k\) event breaks into
    three factors:

    1. Choose which \(k\) of the \(n\) dice show the chosen face:
       \(\binom{n}{k}\).
    2. Make those \(k\) dice land on that face:
       \(\left(\frac{1}{12}\right)^k\).
    3. Make the remaining \(n-k\) dice avoid that face:
       \(\left(\frac{11}{12}\right)^{n-k}\).

    Multiplying those factors gives

    \[
    P(\text{chosen face appears exactly } k \text{ times})
    = \binom{n}{k}\left(\frac{1}{12}\right)^k\left(\frac{11}{12}\right)^{n-k}.
    \]

    If we want "at least \(k\)" instead of "exactly \(k\)," we add the probabilities
    for the exact cases \(k, k+1, \dots, n\).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How likely is one particular set?
    """)
    return


@app.cell(hide_code=True)
def _():
    set_trials = mo.ui.slider(
        steps=TRIAL_STEPS,
        value=20_000,
        label="Set simulation trials",
    )
    set_num_dice = mo.ui.slider(
        start=3,
        stop=9,
        step=1,
        value=5,
        label="Number of d12s",
    )
    set_target_face = mo.ui.slider(
        start=1,
        stop=SIDES,
        step=1,
        value=7,
        label="Chosen face",
    )
    set_match_mode = mo.ui.dropdown(
        options={
            "At least k copies": "at_least",
            "Exactly k copies": "exactly",
        },
        value="At least k copies",
        label="Chosen-face event",
    )
    return set_match_mode, set_num_dice, set_target_face, set_trials


@app.cell(hide_code=True)
def _(set_num_dice):
    scoreable_set_size = mo.ui.slider(
        start=3,
        stop=int(set_num_dice.value),
        step=1,
        value=3,
        label="Set size k",
    )
    return (scoreable_set_size,)


@app.cell(hide_code=True)
def _(
    scoreable_set_size,
    set_match_mode,
    set_num_dice,
    set_target_face,
    set_trials,
):
    mo.vstack(
        [
            mo.md(
                "Use these controls to compare one chosen face with the event that some face forms a set. A 3-set is the first scoreable set in Naasii; larger sets are rarer, but they are worth more because a set scores its size."
            ),
            set_trials,
            set_num_dice,
            scoreable_set_size,
            set_target_face,
            set_match_mode,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    scoreable_set_size,
    set_match_mode,
    set_num_dice,
    set_target_face,
    set_trials,
):
    set_dice_count_value = int(set_num_dice.value)
    set_size_value = int(scoreable_set_size.value)
    set_target_face_value = int(set_target_face.value)
    set_trial_count = int(set_trials.value)
    set_match_mode_value = str(set_match_mode.value)
    set_match_mode_text = "at least" if set_match_mode_value == "at_least" else "exactly"

    set_rolls = roll_d12s(set_trial_count, set_dice_count_value)
    chosen_face_count_values, chosen_face_exact_probabilities = (
        particular_face_count_distribution(set_dice_count_value)
    )
    chosen_face_outcome_counts = np.array(
        [
            math.comb(set_dice_count_value, int(count))
            * (SIDES - 1) ** (set_dice_count_value - int(count))
            for count in chosen_face_count_values
        ],
        dtype=np.int64,
    )
    chosen_face_counts = np.count_nonzero(set_rolls == set_target_face_value, axis=1)
    chosen_face_simulated_probabilities = (
        np.bincount(chosen_face_counts, minlength=set_dice_count_value + 1)
        / set_trial_count
    )

    if set_match_mode_value == "exactly":
        chosen_face_event_mask = chosen_face_count_values == set_size_value
        chosen_face_count_formula = (
            rf"\binom{{{set_dice_count_value}}}{{{set_size_value}}}"
            rf" 11^{{{set_dice_count_value - set_size_value}}}"
        )
        chosen_face_probability_formula = (
            rf"\binom{{{set_dice_count_value}}}{{{set_size_value}}}"
            rf"\left(\frac{{1}}{{12}}\right)^{{{set_size_value}}}"
            rf"\left(\frac{{11}}{{12}}\right)^{{{set_dice_count_value - set_size_value}}}"
        )
    else:
        chosen_face_event_mask = chosen_face_count_values >= set_size_value
        chosen_face_count_formula = (
            rf"\sum_{{j={set_size_value}}}^{{{set_dice_count_value}}}"
            rf"\binom{{{set_dice_count_value}}}{{j}} 11^{{{set_dice_count_value}-j}}"
        )
        chosen_face_probability_formula = (
            rf"\sum_{{j={set_size_value}}}^{{{set_dice_count_value}}}"
            rf"\binom{{{set_dice_count_value}}}{{j}}"
            rf"\left(\frac{{1}}{{12}}\right)^{{j}}"
            rf"\left(\frac{{11}}{{12}}\right)^{{{set_dice_count_value}-j}}"
        )

    chosen_face_exact_probability = float(
        chosen_face_exact_probabilities[chosen_face_event_mask].sum()
    )
    chosen_face_simulated_probability = float(
        chosen_face_simulated_probabilities[chosen_face_event_mask].sum()
    )
    chosen_face_favorable_outcomes = int(
        chosen_face_outcome_counts[chosen_face_event_mask].sum()
    )
    chosen_face_event_label = (
        f"Face {set_target_face_value} appears {set_match_mode_text} {set_size_value} times"
    )
    total_ordered_outcomes = int(SIDES**set_dice_count_value)
    return (
        chosen_face_count_formula,
        chosen_face_count_values,
        chosen_face_event_label,
        chosen_face_event_mask,
        chosen_face_exact_probabilities,
        chosen_face_exact_probability,
        chosen_face_favorable_outcomes,
        chosen_face_probability_formula,
        chosen_face_simulated_probabilities,
        chosen_face_simulated_probability,
        set_dice_count_value,
        set_match_mode_text,
        set_match_mode_value,
        set_rolls,
        set_size_value,
        set_target_face_value,
        set_trial_count,
        total_ordered_outcomes,
    )


@app.cell(hide_code=True)
def _(
    chosen_face_exact_probability,
    set_dice_count_value,
    set_match_mode_text,
    set_match_mode_value,
    set_rolls,
    set_size_value,
):
    set_face_count_matrix = (set_rolls[:, :, None] == np.arange(1, SIDES + 1)).sum(axis=1)
    simulated_largest_set_sizes = set_face_count_matrix.max(axis=1)
    largest_set_sizes, largest_set_exact_probabilities = largest_set_size_distribution(
        set_dice_count_value
    )
    largest_set_simulated_probabilities = (
        np.bincount(simulated_largest_set_sizes, minlength=set_dice_count_value + 1)[
            largest_set_sizes
        ]
        / set_rolls.shape[0]
    )

    if set_match_mode_value == "exactly":
        any_face_event_hits = (set_face_count_matrix == set_size_value).any(axis=1)
    else:
        any_face_event_hits = simulated_largest_set_sizes >= set_size_value

    any_face_event_label = f"Some face appears {set_match_mode_text} {set_size_value} times"
    any_face_exact_probability = exact_any_face_match_probability(
        set_dice_count_value,
        set_size_value,
        match_mode=set_match_mode_value,
    )
    any_face_naive_probability = float(SIDES * chosen_face_exact_probability)
    any_face_overlap_gap = float(any_face_naive_probability - any_face_exact_probability)
    any_face_simulated_probability = float(any_face_event_hits.mean())
    any_face_sample_sizes, any_face_running_rates = running_event_rate(any_face_event_hits)

    exact_scoreable_set_probability = exact_any_set_probability(
        set_dice_count_value, min_size=3
    )
    exact_selected_threshold_probability = exact_any_set_probability(
        set_dice_count_value, min_size=set_size_value
    )
    simulated_scoreable_set_probability = float((simulated_largest_set_sizes >= 3).mean())
    simulated_selected_threshold_probability = float(
        (simulated_largest_set_sizes >= set_size_value).mean()
    )
    return (
        any_face_event_label,
        any_face_exact_probability,
        any_face_naive_probability,
        any_face_overlap_gap,
        any_face_running_rates,
        any_face_sample_sizes,
        any_face_simulated_probability,
        exact_scoreable_set_probability,
        exact_selected_threshold_probability,
        largest_set_exact_probabilities,
        largest_set_simulated_probabilities,
        largest_set_sizes,
        simulated_scoreable_set_probability,
        simulated_selected_threshold_probability,
    )


@app.cell(hide_code=True)
def _(
    chosen_face_count_values,
    chosen_face_event_label,
    chosen_face_event_mask,
    chosen_face_exact_probabilities,
    chosen_face_simulated_probabilities,
    set_dice_count_value,
    set_target_face_value,
):
    chosen_set_fig, chosen_set_ax = plt.subplots(figsize=(10, 4.5))
    chosen_set_ax.bar(
        chosen_face_count_values,
        chosen_face_exact_probabilities,
        width=0.7,
        alpha=0.75,
        color="tab:blue",
        label="Exact probability",
    )
    chosen_set_ax.bar(
        chosen_face_count_values[chosen_face_event_mask],
        chosen_face_exact_probabilities[chosen_face_event_mask],
        width=0.7,
        color="tab:orange",
        label=chosen_face_event_label,
    )
    chosen_set_ax.scatter(
        chosen_face_count_values,
        chosen_face_simulated_probabilities,
        color="black",
        zorder=3,
        label="Simulated frequency",
    )
    chosen_set_ax.set_title(
        f"How many times does face {set_target_face_value} appear among {set_dice_count_value} d12s?"
    )
    chosen_set_ax.set_xlabel(f"Copies of face {set_target_face_value}")
    chosen_set_ax.set_ylabel("Probability")
    chosen_set_ax.set_xticks(chosen_face_count_values)
    chosen_set_ax.grid(axis="y", alpha=0.2)
    chosen_set_ax.legend()
    chosen_set_fig.tight_layout()
    chosen_set_fig
    return


@app.cell(hide_code=True)
def _(
    chosen_face_count_formula,
    chosen_face_event_label,
    chosen_face_exact_probability,
    chosen_face_favorable_outcomes,
    chosen_face_probability_formula,
    chosen_face_simulated_probability,
    set_trial_count,
    total_ordered_outcomes,
):
    mo.md(
        "\n".join(
            [
                f"Counting the event **{chosen_face_event_label.lower()}**:",
                "",
                "| Quantity | Value |",
                "| --- | ---: |",
                f"| Total ordered outcomes | $12^n = {total_ordered_outcomes:,}$ |",
                f"| Favorable ordered outcomes | ${chosen_face_count_formula} = {chosen_face_favorable_outcomes:,}$ |",
                f"| Exact probability formula | ${chosen_face_probability_formula}$ |",
                f"| Exact probability | **{chosen_face_exact_probability:.3%}** |",
                f"| Simulated estimate from {set_trial_count:,} rolls | **{chosen_face_simulated_probability:.3%}** |",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Why can't we just multiply by 12?

    To generalize from one chosen face to "some face," the first instinct is often to
    multiply by \(12\). That is a useful starting point, but it only works when the
    face-specific events are disjoint.

    Once a roll can satisfy the event for more than one face at the same time, simple
    multiplication overcounts. The exact calculation has to count face-frequency
    patterns instead: for each vector \((c_1, \dots, c_{12})\) with
    \(c_1 + \cdots + c_{12} = n\), the ordered outcomes contribute

    \[
    \frac{n!}{c_1!c_2!\cdots c_{12}!}
    \]

    sequences, and we keep only the vectors that make the event true.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How likely is any face to make the set?
    """)
    return


@app.cell(hide_code=True)
def _(
    any_face_event_label,
    any_face_exact_probability,
    any_face_naive_probability,
    any_face_overlap_gap,
    any_face_simulated_probability,
    chosen_face_event_label,
):
    overcount_note = (
        "For this selection, multiplying by 12 happens to be exact because the 12 "
        "face-specific events are disjoint."
    )
    if not np.isclose(any_face_overlap_gap, 0.0):
        overcount_note = (
            "Here multiplying by 12 overcounts because one roll can make more than one "
            "face-specific event true at once."
        )

    mo.md(
        "\n".join(
            [
                f"Comparing **{chosen_face_event_label.lower()}** with **{any_face_event_label.lower()}**:",
                "",
                "| Quantity | Value |",
                "| --- | ---: |",
                f"| Naive multiplier $12 \\times P(\\text{{chosen face event}})$ | **{any_face_naive_probability:.3%}** |",
                f"| Exact probability of {any_face_event_label.lower()} | **{any_face_exact_probability:.3%}** |",
                f"| Simulated estimate of {any_face_event_label.lower()} | **{any_face_simulated_probability:.3%}** |",
                f"| Overcount gap | **{any_face_overlap_gap:.3%}** |",
                "",
                overcount_note,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(
    any_face_event_label,
    any_face_exact_probability,
    any_face_running_rates,
    any_face_sample_sizes,
):
    any_set_conv_fig, any_set_conv_ax = plt.subplots(figsize=(10, 4))
    any_set_conv_ax.plot(
        any_face_sample_sizes,
        any_face_running_rates,
        linewidth=2,
        label="Running simulated estimate",
    )
    any_set_conv_ax.axhline(
        any_face_exact_probability,
        color="black",
        linestyle="--",
        label="Exact probability",
    )
    any_set_conv_ax.set_title(f"Simulation converges for {any_face_event_label.lower()}")
    any_set_conv_ax.set_xlabel("Number of simulated rolls used")
    any_set_conv_ax.set_ylabel("Probability")
    any_set_conv_ax.grid(alpha=0.2)
    any_set_conv_ax.legend()
    any_set_conv_fig.tight_layout()
    any_set_conv_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How often do we make a 3-set, 4-set, or better?

    For Naasii, the key scoring summary is the size of the largest set on the roll. A
    largest set of size \(1\) or \(2\) is not scoreable. A largest set of size \(3\) is
    the first scoreable case, and larger values mean rarer but better-scoring sets.
    """)
    return


@app.cell(hide_code=True)
def _(
    largest_set_exact_probabilities,
    largest_set_simulated_probabilities,
    largest_set_sizes,
):
    scoreable_mask = largest_set_sizes >= 3
    largest_set_fig, largest_set_ax = plt.subplots(figsize=(10, 4.5))
    largest_set_ax.axvspan(0.5, 2.5, color="tab:gray", alpha=0.08)
    largest_set_ax.axvspan(2.5, largest_set_sizes[-1] + 0.5, color="tab:orange", alpha=0.06)
    largest_set_ax.bar(
        largest_set_sizes[~scoreable_mask],
        largest_set_exact_probabilities[~scoreable_mask],
        width=0.7,
        color="tab:gray",
        alpha=0.85,
        label="Exact probability (not scoreable)",
    )
    largest_set_ax.bar(
        largest_set_sizes[scoreable_mask],
        largest_set_exact_probabilities[scoreable_mask],
        width=0.7,
        color="tab:orange",
        alpha=0.85,
        label="Exact probability (scoreable)",
    )
    largest_set_ax.scatter(
        largest_set_sizes,
        largest_set_simulated_probabilities,
        color="black",
        zorder=3,
        label="Simulated frequency",
    )
    largest_set_ax.axvline(2.5, color="black", linestyle="--", linewidth=1)
    largest_set_ax.set_title("Distribution of the largest set size on the roll")
    largest_set_ax.set_xlabel("Largest set size")
    largest_set_ax.set_ylabel("Probability")
    largest_set_ax.set_xticks(largest_set_sizes)
    largest_set_ax.grid(axis="y", alpha=0.2)
    largest_set_ax.legend()
    largest_set_fig.tight_layout()
    largest_set_fig
    return


@app.cell(hide_code=True)
def _(
    exact_scoreable_set_probability,
    exact_selected_threshold_probability,
    largest_set_exact_probabilities,
    largest_set_simulated_probabilities,
    largest_set_sizes,
    set_dice_count_value,
    set_size_value,
    simulated_scoreable_set_probability,
    simulated_selected_threshold_probability,
):
    no_scoreable_exact_probability = float(
        largest_set_exact_probabilities[largest_set_sizes < 3].sum()
    )
    no_scoreable_simulated_probability = float(
        largest_set_simulated_probabilities[largest_set_sizes < 3].sum()
    )
    score_rows = [
        "| Outcome | Exact probability | Simulated probability | Naasii meaning |",
        "| --- | ---: | ---: | --- |",
        (
            f"| No scoreable set (largest set at most 2) | "
            f"{no_scoreable_exact_probability:.3%} | "
            f"{no_scoreable_simulated_probability:.3%} | "
            "No set score |"
        ),
    ]
    for largest_size, exact_probability, simulated_probability in zip(
        largest_set_sizes[largest_set_sizes >= 3],
        largest_set_exact_probabilities[largest_set_sizes >= 3],
        largest_set_simulated_probabilities[largest_set_sizes >= 3],
    ):
        score_rows.append(
            f"| Largest set = {int(largest_size)} | "
            f"{exact_probability:.3%} | "
            f"{simulated_probability:.3%} | "
            f"{int(largest_size)}-point set |"
        )

    mo.md(
        "\n".join(
            [
                f"With **{set_dice_count_value} d12s**, the exact probability of **some scoreable set (3+)** is **{exact_scoreable_set_probability:.3%}** and the simulated estimate is **{simulated_scoreable_set_probability:.3%}**.",
                f"If you raise the threshold to **{set_size_value}+**, the exact probability becomes **{exact_selected_threshold_probability:.3%}** and the simulated estimate is **{simulated_selected_threshold_probability:.3%}**.",
                "",
                *score_rows,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(
    exact_scoreable_set_probability,
    exact_selected_threshold_probability,
    set_dice_count_value,
    set_size_value,
):
    threshold_sentence = ""
    if set_size_value > 3:
        threshold_sentence = (
            f" Requiring at least **{set_size_value}** of a kind cuts that probability "
            f"to **{exact_selected_threshold_probability:.3%}**."
        )

    mo.md(
        f"""
        For **{set_dice_count_value} fair d12s**, a scoreable set appears on about
        **{exact_scoreable_set_probability:.3%}** of rolls.{threshold_sentence}

        The graph and table above show the main Naasii lesson: three of a kind is the
        entry point for scoring, but larger sets are much rarer even though they are
        worth more points.

        Sets are only half of the scoring story. The next question is how often several
        d12s make a run of consecutive values.
        """
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## When do several d12s make a scoreable run?

    In Naasii, a **run** means consecutive values such as \(4, 5, 6\). A run becomes
    scoreable at length \(3\), and longer runs are worth more points. Unlike sets,
    repeated faces do not extend the run: the roll \((2, 2, 3, 4)\) still has longest
    run \(3\), not \(4\), because its distinct values are just \(2, 3, 4\).
    """)
    return


@app.cell
def _():
    run_example_rolls = (
        (2, 2, 7, 9, 12),
        (1, 2, 5, 8, 9),
        (2, 2, 3, 4, 9),
        (7, 8, 8, 9, 10),
    )
    run_example_rows = "\n".join(
        (
            f"| ({', '.join(str(value) for value in _example_roll)}) | "
            f"{', '.join(str(value) for value in np.unique(_example_roll))} | "
            f"{longest_run_length(_example_roll)} | "
            f"{'Yes' if longest_run_length(_example_roll) >= 3 else 'No'} |"
        )
        for _example_roll in run_example_rolls
    )
    mo.md(
        "\n".join(
            [
                "A few five-die examples make the rule concrete:",
                "",
                "| Roll | Distinct faces present | Longest run | Scoreable run? |",
                "| --- | --- | ---: | --- |",
                run_example_rows,
            ]
        )
    )
    return


@app.cell
def _():
    mo.md(r"""
    ### Why are runs harder to count than sets?

    For sets, the face counts are enough: if one face appears \(k\) times, we have the
    event. Runs are more delicate because **adjacency** matters. The rolls
    \((2, 2, 3, 4, 9)\) and \((2, 2, 3, 5, 6)\) have the same multiplicities
    \((2, 1, 1, 1)\), but only the first contains a 3-run.

    So the exact run calculation still groups outcomes by face-count vectors, but then
    it asks a new question: which faces are present, and what is the longest
    consecutive block among those present faces?
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### How likely is a scoreable run?
    """)
    return


@app.cell
def _():
    run_trials = mo.ui.slider(
        steps=TRIAL_STEPS,
        value=20_000,
        label="Run simulation trials",
    )
    run_num_dice = mo.ui.slider(
        start=3,
        stop=9,
        step=1,
        value=5,
        label="Number of d12s",
    )
    run_match_mode = mo.ui.dropdown(
        options={
            "At least k in a run": "at_least",
            "Exactly k as the longest run": "exactly",
        },
        value="At least k in a run",
        label="Run event",
    )
    return run_match_mode, run_num_dice, run_trials


@app.cell
def _(run_num_dice):
    scoreable_run_size = mo.ui.slider(
        start=3,
        stop=int(run_num_dice.value),
        step=1,
        value=3,
        label="Run size k",
    )
    return (scoreable_run_size,)


@app.cell
def _(run_match_mode, run_num_dice, run_trials, scoreable_run_size):
    mo.vstack(
        [
            mo.md(
                "Use these controls to compare the first scoreable run size, 3, with stricter run targets. The exact model tracks the longest block of consecutive distinct faces, while the simulation estimates the same probability from many random rolls."
            ),
            run_trials,
            run_num_dice,
            scoreable_run_size,
            run_match_mode,
        ]
    )
    return


@app.cell
def _(run_match_mode, run_num_dice, run_trials, scoreable_run_size):
    run_dice_count_value = int(run_num_dice.value)
    run_size_value = int(scoreable_run_size.value)
    run_trial_count = int(run_trials.value)
    run_match_mode_value = str(run_match_mode.value)

    run_rolls = roll_d12s(run_trial_count, run_dice_count_value)
    run_length_values, largest_run_exact_probabilities = largest_run_length_distribution(
        run_dice_count_value
    )
    largest_run_outcome_counts = largest_run_length_outcome_counts(run_dice_count_value)[
        run_length_values
    ]
    run_face_presence = (run_rolls[:, :, None] == np.arange(1, SIDES + 1)).any(axis=1)
    run_streak_lengths = run_face_presence.astype(np.int64)

    for _face_index in range(1, SIDES):
        run_streak_lengths[:, _face_index] = np.where(
            run_face_presence[:, _face_index],
            run_streak_lengths[:, _face_index - 1] + 1,
            0,
        )

    simulated_largest_run_lengths = run_streak_lengths.max(axis=1)
    largest_run_simulated_probabilities = (
        np.bincount(simulated_largest_run_lengths, minlength=run_dice_count_value + 1)[
            run_length_values
        ]
        / run_trial_count
    )

    if run_match_mode_value == "exactly":
        run_event_mask = run_length_values == run_size_value
        run_event_hits = simulated_largest_run_lengths == run_size_value
        selected_run_event_label = f"The largest run has length exactly {run_size_value}"
    else:
        run_event_mask = run_length_values >= run_size_value
        run_event_hits = simulated_largest_run_lengths >= run_size_value
        selected_run_event_label = f"Some run has length at least {run_size_value}"

    selected_run_exact_probability = float(
        largest_run_exact_probabilities[run_event_mask].sum()
    )
    selected_run_favorable_outcomes = int(largest_run_outcome_counts[run_event_mask].sum())
    selected_run_simulated_probability = float(run_event_hits.mean())
    run_sample_sizes, run_running_rates = running_event_rate(run_event_hits)
    exact_scoreable_run_probability = exact_any_run_probability(
        run_dice_count_value, min_length=3
    )
    simulated_scoreable_run_probability = float((simulated_largest_run_lengths >= 3).mean())
    total_run_outcomes = int(SIDES**run_dice_count_value)
    return (
        exact_scoreable_run_probability,
        largest_run_exact_probabilities,
        largest_run_simulated_probabilities,
        run_dice_count_value,
        run_length_values,
        run_running_rates,
        run_sample_sizes,
        run_trial_count,
        selected_run_event_label,
        selected_run_exact_probability,
        selected_run_favorable_outcomes,
        selected_run_simulated_probability,
        simulated_scoreable_run_probability,
        total_run_outcomes,
    )


@app.cell
def _(
    largest_run_exact_probabilities,
    largest_run_simulated_probabilities,
    run_length_values,
):
    run_scoreable_mask = run_length_values >= 3
    largest_run_fig, largest_run_ax = plt.subplots(figsize=(10, 4.5))
    largest_run_ax.axvspan(0.5, 2.5, color="tab:gray", alpha=0.08)
    largest_run_ax.axvspan(2.5, run_length_values[-1] + 0.5, color="tab:green", alpha=0.06)
    largest_run_ax.bar(
        run_length_values[~run_scoreable_mask],
        largest_run_exact_probabilities[~run_scoreable_mask],
        width=0.7,
        color="tab:gray",
        alpha=0.85,
        label="Exact probability (not scoreable)",
    )
    largest_run_ax.bar(
        run_length_values[run_scoreable_mask],
        largest_run_exact_probabilities[run_scoreable_mask],
        width=0.7,
        color="tab:green",
        alpha=0.85,
        label="Exact probability (scoreable)",
    )
    largest_run_ax.scatter(
        run_length_values,
        largest_run_simulated_probabilities,
        color="black",
        zorder=3,
        label="Simulated frequency",
    )
    largest_run_ax.axvline(2.5, color="black", linestyle="--", linewidth=1)
    largest_run_ax.set_title("Distribution of the largest run length on the roll")
    largest_run_ax.set_xlabel("Largest run length")
    largest_run_ax.set_ylabel("Probability")
    largest_run_ax.set_xticks(run_length_values)
    largest_run_ax.grid(axis="y", alpha=0.2)
    largest_run_ax.legend()
    largest_run_fig.tight_layout()
    largest_run_fig
    return


@app.cell
def _(
    run_trial_count,
    selected_run_event_label,
    selected_run_exact_probability,
    selected_run_favorable_outcomes,
    selected_run_simulated_probability,
    total_run_outcomes,
):
    mo.md(
        "\n".join(
            [
                f"Counting the event **{selected_run_event_label.lower()}**:",
                "",
                "| Quantity | Value |",
                "| --- | ---: |",
                f"| Total ordered outcomes | $12^n = {total_run_outcomes:,}$ |",
                f"| Favorable ordered outcomes | **{selected_run_favorable_outcomes:,}** |",
                f"| Exact probability | **{selected_run_exact_probability:.3%}** |",
                f"| Simulated estimate from {run_trial_count:,} rolls | **{selected_run_simulated_probability:.3%}** |",
                "",
                "There is no single shortcut like \\(\\binom{n}{k}\\) here, because different patterns of repeated and missing faces can still lead to the same longest run length.",
            ]
        )
    )
    return


@app.cell
def _(
    run_running_rates,
    run_sample_sizes,
    selected_run_event_label,
    selected_run_exact_probability,
):
    run_convergence_fig, run_convergence_ax = plt.subplots(figsize=(10, 4))
    run_convergence_ax.plot(
        run_sample_sizes,
        run_running_rates,
        linewidth=2,
        label="Running simulated estimate",
    )
    run_convergence_ax.axhline(
        selected_run_exact_probability,
        color="black",
        linestyle="--",
        label="Exact probability",
    )
    run_convergence_ax.set_title(
        f"Simulation converges for {selected_run_event_label.lower()}"
    )
    run_convergence_ax.set_xlabel("Number of simulated rolls used")
    run_convergence_ax.set_ylabel("Probability")
    run_convergence_ax.grid(alpha=0.2)
    run_convergence_ax.legend()
    run_convergence_fig.tight_layout()
    run_convergence_fig
    return


@app.cell
def _(
    exact_scoreable_run_probability,
    largest_run_exact_probabilities,
    largest_run_simulated_probabilities,
    run_dice_count_value,
    run_length_values,
    selected_run_event_label,
    selected_run_exact_probability,
    selected_run_simulated_probability,
    simulated_scoreable_run_probability,
):
    no_scoreable_run_exact_probability = float(
        largest_run_exact_probabilities[run_length_values < 3].sum()
    )
    no_scoreable_run_simulated_probability = float(
        largest_run_simulated_probabilities[run_length_values < 3].sum()
    )
    run_rows = [
        "| Outcome | Exact probability | Simulated probability | Naasii meaning |",
        "| --- | ---: | ---: | --- |",
        (
            f"| No scoreable run (largest run at most 2) | "
            f"{no_scoreable_run_exact_probability:.3%} | "
            f"{no_scoreable_run_simulated_probability:.3%} | "
            "No run score |"
        ),
    ]
    for _largest_run_size, _exact_probability, _simulated_probability in zip(
        run_length_values[run_length_values >= 3],
        largest_run_exact_probabilities[run_length_values >= 3],
        largest_run_simulated_probabilities[run_length_values >= 3],
    ):
        run_rows.append(
            f"| Largest run = {int(_largest_run_size)} | "
            f"{_exact_probability:.3%} | "
            f"{_simulated_probability:.3%} | "
            f"{int(_largest_run_size)}-point run |"
        )

    mo.md(
        "\n".join(
            [
                f"With **{run_dice_count_value} d12s**, the exact probability of **some scoreable run (3+)** is **{exact_scoreable_run_probability:.3%}** and the simulated estimate is **{simulated_scoreable_run_probability:.3%}**.",
                f"For the selected event **{selected_run_event_label.lower()}**, the exact probability is **{selected_run_exact_probability:.3%}** and the simulated estimate is **{selected_run_simulated_probability:.3%}**.",
                "",
                *run_rows,
            ]
        )
    )
    return


@app.cell
def _(
    exact_scoreable_run_probability,
    run_dice_count_value,
    selected_run_event_label,
    selected_run_exact_probability,
):
    mo.md(
        f"""
        For **{run_dice_count_value} fair d12s**, a scoreable run appears on about
        **{exact_scoreable_run_probability:.3%}** of rolls.

        Runs reward a different kind of luck than sets. Duplicates can still matter, but
        only indirectly: they help only if enough neighboring values also appear to make
        a long consecutive block. For the selected event
        **{selected_run_event_label.lower()}**, the exact probability is
        **{selected_run_exact_probability:.3%}**.

        That finishes the ordinary fair-d12 model for the two basic scoring patterns.
        The next step is to ask how Naasii's special dice and values change those
        probabilities.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## After that: How do Naasii's special dice and values change the model?

    Once sets and runs are understood for ordinary fair d12s, the model can move closer
    to the actual game by adding the distinctions between dice types and any special
    values, such as a wild value.
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
def particular_face_count_distribution(
    num_dice: int, sides: int = SIDES
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact distribution for how often one chosen face appears."""
    if num_dice < 0:
        raise ValueError("num_dice must be nonnegative")
    if sides <= 0:
        raise ValueError("sides must be positive")

    count_values = np.arange(num_dice + 1, dtype=np.int64)
    if sides == 1:
        probabilities = np.zeros(num_dice + 1, dtype=float)
        probabilities[-1] = 1.0
        return count_values, probabilities

    total_outcomes = sides**num_dice
    probabilities = np.array(
        [
            math.comb(num_dice, int(count))
            * (sides - 1) ** (num_dice - int(count))
            / total_outcomes
            for count in count_values
        ],
        dtype=float,
    )
    return count_values, probabilities


@app.function(hide_code=True)
def enumerate_position_choices(
    num_dice: int, choose_count: int
) -> tuple[tuple[int, ...], ...]:
    """List the die-position choices counted by n choose k."""
    if num_dice < 0:
        raise ValueError("num_dice must be nonnegative")
    if choose_count < 0 or choose_count > num_dice:
        raise ValueError("choose_count must satisfy 0 <= choose_count <= num_dice")

    from itertools import combinations

    return tuple(combinations(range(1, num_dice + 1), choose_count))


@app.function(hide_code=True)
def iterate_face_count_vectors(num_dice: int, sides: int = SIDES):
    """Yield labeled face-count vectors whose entries sum to num_dice."""
    if num_dice < 0:
        raise ValueError("num_dice must be nonnegative")
    if sides <= 0:
        raise ValueError("sides must be positive")

    current = [0] * sides

    def _recurse(face_index: int, remaining: int):
        if face_index == sides - 1:
            current[face_index] = remaining
            yield tuple(current)
            return

        for face_count in range(remaining + 1):
            current[face_index] = face_count
            yield from _recurse(face_index + 1, remaining - face_count)

    yield from _recurse(0, num_dice)


@app.function(hide_code=True)
def largest_set_size_distribution(
    num_dice: int, sides: int = SIDES
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact distribution of the largest multiplicity on a roll."""
    if num_dice < 0:
        raise ValueError("num_dice must be nonnegative")
    if sides <= 0:
        raise ValueError("sides must be positive")
    if num_dice == 0:
        return np.array([0], dtype=np.int64), np.array([1.0])

    factorials = [math.factorial(i) for i in range(num_dice + 1)]
    total_permutations = factorials[num_dice]
    largest_size_outcome_counts = np.zeros(num_dice + 1, dtype=np.int64)

    for count_vector in iterate_face_count_vectors(num_dice, sides):
        outcome_count = total_permutations
        for count in count_vector:
            outcome_count //= factorials[count]
        largest_size_outcome_counts[max(count_vector)] += outcome_count

    largest_sizes = np.arange(1, num_dice + 1, dtype=np.int64)
    probabilities = largest_size_outcome_counts[largest_sizes] / (sides**num_dice)
    return largest_sizes, probabilities


@app.function(hide_code=True)
def exact_any_set_probability(
    num_dice: int, min_size: int = 3, sides: int = SIDES
) -> float:
    """Compute the probability that some face appears at least min_size times."""
    if min_size <= 0:
        return 1.0

    largest_sizes, largest_probabilities = largest_set_size_distribution(
        num_dice, sides=sides
    )
    return float(largest_probabilities[largest_sizes >= min_size].sum())


@app.function(hide_code=True)
def exact_any_face_match_probability(
    num_dice: int,
    set_size: int,
    match_mode: str = "at_least",
    sides: int = SIDES,
) -> float:
    """Compute the probability that some face matches the selected multiplicity event."""
    if num_dice < 0:
        raise ValueError("num_dice must be nonnegative")
    if sides <= 0:
        raise ValueError("sides must be positive")
    if match_mode not in {"at_least", "exactly"}:
        raise ValueError("match_mode must be 'at_least' or 'exactly'")
    if set_size <= 0:
        return 1.0
    if set_size > num_dice:
        return 0.0

    factorials = [math.factorial(i) for i in range(num_dice + 1)]
    total_permutations = factorials[num_dice]
    favorable_outcomes = 0

    for count_vector in iterate_face_count_vectors(num_dice, sides):
        if match_mode == "exactly":
            event_holds = set_size in count_vector
        else:
            event_holds = max(count_vector) >= set_size

        if not event_holds:
            continue

        outcome_count = total_permutations
        for count in count_vector:
            outcome_count //= factorials[count]
        favorable_outcomes += outcome_count

    return float(favorable_outcomes / (sides**num_dice))


@app.function
def longest_run_length(roll: np.ndarray) -> int:
    """Return the largest number of consecutive distinct faces in a roll."""
    roll_values = np.asarray(roll, dtype=np.int64)
    if roll_values.ndim != 1:
        raise ValueError("roll must be one-dimensional")
    if roll_values.size == 0:
        return 0

    distinct_faces = np.unique(roll_values)
    longest_length = 1
    current_length = 1

    for previous_face, current_face in zip(distinct_faces[:-1], distinct_faces[1:]):
        if current_face == previous_face + 1:
            current_length += 1
            longest_length = max(longest_length, current_length)
        else:
            current_length = 1

    return int(longest_length)


@app.function
def longest_true_streak(values: np.ndarray) -> int:
    """Return the largest number of consecutive True values."""
    truth_values = np.asarray(values, dtype=bool)
    if truth_values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if truth_values.size == 0:
        return 0

    longest_length = 0
    current_length = 0

    for is_present in truth_values:
        if is_present:
            current_length += 1
            longest_length = max(longest_length, current_length)
        else:
            current_length = 0

    return int(longest_length)


@app.function
def largest_run_length_outcome_counts(num_dice: int, sides: int = SIDES) -> np.ndarray:
    """Count ordered outcomes by the largest run length present."""
    if num_dice < 0:
        raise ValueError("num_dice must be nonnegative")
    if sides <= 0:
        raise ValueError("sides must be positive")
    if num_dice == 0:
        return np.array([1], dtype=np.int64)

    factorials = [math.factorial(i) for i in range(num_dice + 1)]
    total_permutations = factorials[num_dice]
    longest_run_outcome_counts = np.zeros(num_dice + 1, dtype=np.int64)

    for count_vector in iterate_face_count_vectors(num_dice, sides):
        outcome_count = total_permutations
        for count in count_vector:
            outcome_count //= factorials[count]
        longest_run_outcome_counts[
            longest_true_streak(np.array(count_vector, dtype=np.int64) > 0)
        ] += outcome_count

    return longest_run_outcome_counts


@app.function
def largest_run_length_distribution(
    num_dice: int, sides: int = SIDES
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact distribution of the largest run length on a roll."""
    outcome_counts = largest_run_length_outcome_counts(num_dice, sides=sides)
    if num_dice == 0:
        return np.array([0], dtype=np.int64), np.array([1.0])

    run_lengths = np.arange(1, num_dice + 1, dtype=np.int64)
    probabilities = outcome_counts[run_lengths] / (sides**num_dice)
    return run_lengths, probabilities


@app.function
def exact_any_run_probability(
    num_dice: int, min_length: int = 3, sides: int = SIDES
) -> float:
    """Compute the probability that some run has length at least min_length."""
    if min_length <= 0:
        return 1.0
    if min_length > num_dice:
        return 0.0

    run_lengths, run_probabilities = largest_run_length_distribution(
        num_dice, sides=sides
    )
    return float(run_probabilities[run_lengths >= min_length].sum())


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
    def brute_force_largest_set_size_distribution(num_dice: int) -> np.ndarray:
        from itertools import product

        outcome_counts = np.zeros(num_dice + 1, dtype=np.int64)
        for outcome in product(range(1, SIDES + 1), repeat=num_dice):
            _, face_counts = np.unique(outcome, return_counts=True)
            outcome_counts[face_counts.max()] += 1
        return outcome_counts / SIDES**num_dice


    def brute_force_any_face_match_probability(
        num_dice: int, set_size: int, match_mode: str
    ) -> float:
        from itertools import product

        hits = 0
        for outcome in product(range(1, SIDES + 1), repeat=num_dice):
            _, face_counts = np.unique(outcome, return_counts=True)
            if match_mode == "exactly":
                event_holds = np.any(face_counts == set_size)
            else:
                event_holds = face_counts.max() >= set_size
            hits += int(event_holds)
        return hits / SIDES**num_dice

    def brute_force_largest_run_length_distribution(num_dice: int) -> np.ndarray:
        from itertools import product

        outcome_counts = np.zeros(num_dice + 1, dtype=np.int64)
        for outcome in product(range(1, SIDES + 1), repeat=num_dice):
            outcome_counts[longest_run_length(np.array(outcome, dtype=np.int64))] += 1
        return outcome_counts / SIDES**num_dice

    def brute_force_any_run_probability(num_dice: int, min_length: int) -> float:
        from itertools import product

        hits = 0
        for outcome in product(range(1, SIDES + 1), repeat=num_dice):
            hits += int(
                longest_run_length(np.array(outcome, dtype=np.int64)) >= min_length
            )
        return hits / SIDES**num_dice

    return (
        brute_force_any_face_match_probability,
        brute_force_any_run_probability,
        brute_force_largest_run_length_distribution,
        brute_force_largest_set_size_distribution,
    )


@app.cell
def _(
    brute_force_any_face_match_probability,
    brute_force_any_run_probability,
    brute_force_largest_run_length_distribution,
    brute_force_largest_set_size_distribution,
):
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

        with patch.object(
            np.random, "default_rng", side_effect=[rng_a, rng_b]
        ) as default_rng:
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


    def test_particular_face_count_distribution_three_dice_known_values():
        count_values, probabilities = particular_face_count_distribution(num_dice=3)

        expected_probabilities = np.array([11**3, 3 * 11**2, 3 * 11, 1]) / SIDES**3
        assert np.array_equal(count_values, np.arange(4))
        assert np.allclose(probabilities, expected_probabilities)


    def test_particular_face_count_distribution_probabilities_sum_to_one():
        count_values, probabilities = particular_face_count_distribution(num_dice=5)

        assert np.array_equal(count_values, np.arange(6))
        assert np.isclose(probabilities.sum(), 1.0)


    def test_enumerate_position_choices_matches_comb_count():
        position_choices = enumerate_position_choices(5, 3)

        assert len(position_choices) == math.comb(5, 3)
        assert position_choices[0] == (1, 2, 3)
        assert position_choices[-1] == (3, 4, 5)


    def test_enumerate_position_choices_edge_cases():
        assert enumerate_position_choices(4, 0) == ((),)
        assert enumerate_position_choices(4, 4) == ((1, 2, 3, 4),)


    def test_enumerate_position_choices_rejects_invalid_k():
        for invalid_k in (-1, 5):
            try:
                enumerate_position_choices(4, invalid_k)
            except ValueError as exc:
                assert str(exc) == "choose_count must satisfy 0 <= choose_count <= num_dice"
            else:
                assert False, "Expected ValueError for invalid choose_count"

    def test_longest_run_length_counts_consecutive_distinct_faces():
        assert longest_run_length(np.array([2, 3, 4, 9])) == 3


    def test_longest_run_length_ignores_duplicates():
        assert longest_run_length(np.array([2, 2, 3, 4])) == 3


    def test_longest_run_length_all_same_anchor():
        assert longest_run_length(np.array([8, 8, 8, 8])) == 1


    def test_largest_run_length_distribution_matches_bruteforce_three_dice():
        run_lengths, probabilities = largest_run_length_distribution(num_dice=3)

        assert np.array_equal(run_lengths, np.array([1, 2, 3]))
        assert np.allclose(
            probabilities,
            brute_force_largest_run_length_distribution(num_dice=3)[run_lengths],
        )


    def test_largest_run_length_distribution_matches_bruteforce_four_dice():
        run_lengths, probabilities = largest_run_length_distribution(num_dice=4)

        assert np.array_equal(run_lengths, np.array([1, 2, 3, 4]))
        assert np.allclose(
            probabilities,
            brute_force_largest_run_length_distribution(num_dice=4)[run_lengths],
        )


    def test_largest_run_length_distribution_probabilities_sum_to_one():
        _, probabilities = largest_run_length_distribution(num_dice=5)

        assert np.isclose(probabilities.sum(), 1.0)


    def test_exact_any_run_probability_three_dice_scoreable_anchor():
        probability = exact_any_run_probability(num_dice=3, min_length=3)

        assert np.isclose(probability, 10 * math.factorial(3) / SIDES**3)


    def test_exact_any_run_probability_zero_when_threshold_exceeds_num_dice():
        probability = exact_any_run_probability(num_dice=4, min_length=5)

        assert probability == 0.0


    def test_exact_any_run_probability_matches_bruteforce():
        probability = exact_any_run_probability(num_dice=4, min_length=3)

        assert np.isclose(probability, brute_force_any_run_probability(4, 3))


    def test_largest_set_size_distribution_matches_bruteforce_three_dice():
        largest_sizes, probabilities = largest_set_size_distribution(num_dice=3)

        assert np.array_equal(largest_sizes, np.array([1, 2, 3]))
        assert np.allclose(
            probabilities,
            brute_force_largest_set_size_distribution(num_dice=3)[largest_sizes],
        )


    def test_largest_set_size_distribution_matches_bruteforce_four_dice():
        largest_sizes, probabilities = largest_set_size_distribution(num_dice=4)

        assert np.array_equal(largest_sizes, np.array([1, 2, 3, 4]))
        assert np.allclose(
            probabilities,
            brute_force_largest_set_size_distribution(num_dice=4)[largest_sizes],
        )


    def test_largest_set_size_distribution_all_same_anchor():
        largest_sizes, probabilities = largest_set_size_distribution(num_dice=5)

        assert largest_sizes[-1] == 5
        assert np.isclose(probabilities[-1], 1 / SIDES**4)


    def test_exact_any_set_probability_three_dice_scoreable_anchor():
        probability = exact_any_set_probability(num_dice=3, min_size=3)

        assert np.isclose(probability, 1 / SIDES**2)


    def test_exact_any_set_probability_zero_when_threshold_exceeds_num_dice():
        probability = exact_any_set_probability(num_dice=4, min_size=5)

        assert probability == 0.0


    def test_exact_any_face_match_probability_exactly_matches_bruteforce():
        probability = exact_any_face_match_probability(
            num_dice=4, set_size=2, match_mode="exactly"
        )

        assert np.isclose(
            probability,
            brute_force_any_face_match_probability(4, 2, "exactly"),
        )


    def test_exact_any_face_match_probability_at_least_matches_any_set_probability():
        probability = exact_any_face_match_probability(
            num_dice=6, set_size=3, match_mode="at_least"
        )

        assert np.isclose(probability, exact_any_set_probability(6, min_size=3))


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
