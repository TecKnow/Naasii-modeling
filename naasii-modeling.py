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

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    SIDES = 12

    def roll_d12s(rng: np.random.Generator, trials: int, num_dice: int) -> np.ndarray:
        return rng.integers(1, SIDES + 1, size=(trials, num_dice))


    def exact_sum_distribution(num_dice: int, sides: int = SIDES) -> tuple[np.ndarray, np.ndarray]:
        counts = np.ones(sides, dtype=np.int64)
        for _ in range(1, num_dice):
            counts = np.convolve(counts, np.ones(sides, dtype=np.int64))

        totals = np.arange(num_dice, num_dice * sides + 1)
        probabilities = counts / counts.sum()
        return totals, probabilities


    def running_event_rate(event_hits: np.ndarray, points: int = 30) -> tuple[np.ndarray, np.ndarray]:
        sample_sizes = np.unique(
            np.linspace(1, event_hits.size, num=min(points, event_hits.size), dtype=int)
        )
        cumulative_hits = np.cumsum(event_hits.astype(np.int64))
        return sample_sizes, cumulative_hits[sample_sizes - 1] / sample_sizes


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        # Naasii modeling

        This notebook starts with the statistics basics we will need later for Naasii:
        exact probability distributions, simulation with NumPy, and plots that react to
        a few Marimo controls.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Why start here?

        Naasii uses twelve-sided dice, so the examples below use fair d12s. The goal of
        this first pass is not to model the game rules yet. It is to verify the notebook
        workflow and build intuition for:

        - exact probability vs simulated probability
        - how sample size affects stability
        - how to define events that we will later reuse for Naasii decisions

        Later increments can replace these generic events with Naasii-specific ones such
        as making a set, making a run, or surviving Crow dice.
        """
    )
    return


@app.cell
def _():
    trials = mo.ui.slider(
        start=1_000,
        stop=100_000,
        step=1_000,
        value=20_000,
        label="Simulation trials",
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
            mo.md("## Explore a simple d12 model"),
            mo.md(
                "Adjust the controls, then compare exact results to a Monte Carlo simulation."
            ),
            trials,
            num_dice,
            target_sum,
        ]
    )
    return num_dice, target_sum, trials


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Exact distribution and Monte Carlo estimate

        For the selected number of d12s, we compute the exact distribution of the total by
        convolving uniform die outcomes. Then we simulate many rolls with NumPy and check
        how closely the simulation matches the exact result.
        """
    )
    return


@app.cell
def _(num_dice, target_sum, trials):
    rng = np.random.default_rng(2026)
    effective_target = int(np.clip(target_sum.value, num_dice.value, num_dice.value * SIDES))
    rolls = roll_d12s(rng, trials.value, num_dice.value)
    totals = rolls.sum(axis=1)

    possible_totals, exact_probabilities = exact_sum_distribution(num_dice.value)
    simulated_counts = np.bincount(totals, minlength=possible_totals[-1] + 1)
    simulated_probabilities = simulated_counts[possible_totals] / trials.value

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


@app.cell
def _(exact_probabilities, possible_totals, simulated_probabilities):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(
        possible_totals - 0.2,
        exact_probabilities,
        width=0.4,
        label="Exact probability",
    )
    ax.bar(
        possible_totals + 0.2,
        simulated_probabilities,
        width=0.4,
        alpha=0.75,
        label="Simulated probability",
    )
    ax.set_title("Distribution of the total")
    ax.set_xlabel("Total rolled")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(
    effective_target,
    exact_event_probability,
    exact_expected_total,
    num_dice,
    simulated_event_probability,
    simulated_expected_total,
    target_sum,
    totals,
    trials,
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

        With **{num_dice.value} d12s** and **{trials.value:,} simulated rolls**:

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


@app.cell
def _(effective_target, event_running_rates, event_sample_sizes, exact_event_probability):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        event_sample_sizes,
        event_running_rates,
        linewidth=2,
        label="Running simulated estimate",
    )
    ax.axhline(
        exact_event_probability,
        color="black",
        linestyle="--",
        label=f"Exact P(total >= {effective_target})",
    )
    ax.set_title("Simulation converges toward the exact probability")
    ax.set_xlabel("Number of simulated rolls used")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Game resources

        - Official website: https://coyoteandcrow.net/board-game-resources/#Naasii
        - Rules: https://coyoteandcrow.net/wp-content/uploads/2025/12/Naasii-Rules-3.0.pdf
        - Scorecard: https://coyoteandcrow.net/wp-content/uploads/2023/10/Naasii-Scorecard.pdf

        ## To edit

        ```bash
        uv run --with marimo[recommended] marimo edit --sandbox naasii-modeling.py
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
