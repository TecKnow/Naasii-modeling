import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Naasii-modeling
    Modeling the probabilities of the dice game Naasii by Coyote & Crow games
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Naasii
    **Naasii** is a push-your-luck dice game for 2-5 players that takes about an hour to play.  LIke the Coyote & Crow TTRPG, Naasii is based on 12-sided dice divided into two groups.  Naasii uses 9 white Coyote dice and 3 black Crow dice.

    Players attempt to form sets of at least three of the same number or runs of at least three sequential numbers across multiple rolls on their turn.  Each roll after the first provides additional white Coyote dice, but also adds a black Crow die that can cancel dice or even cause the player to bust, ending their turn early with no score.  Players may end their turn after any roll where they can score.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Game resources
    - The game's official website: https://coyoteandcrow.net/board-game-resources/#Naasii
    - The rules: https://coyoteandcrow.net/wp-content/uploads/2025/12/Naasii-Rules-3.0.pdf
    - The scoring sheet: https://coyoteandcrow.net/wp-content/uploads/2023/10/Naasii-Scorecard.pdf
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Marimo notebook
    ### To edit
    ```bash
    uv run --with marimo[recommended] marimo edit naasii-modeling.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
