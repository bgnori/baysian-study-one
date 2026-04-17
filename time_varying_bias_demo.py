from src.bayesian_dice.analysis import detect_bias_change
from src.bayesian_dice.simulator import ChangingBiasDie


def main() -> None:
    die = ChangingBiasDie(
        [
            (30, [0.35, 0.15, 0.15, 0.15, 0.10, 0.10]),
            (30, [0.05, 0.10, 0.10, 0.15, 0.20, 0.40]),
        ],
        seed=41,
    )
    observations = die.sample(60)
    summary = detect_bias_change(observations, window_size=12, step_size=4)

    print("Advanced demo: detecting a time-varying hidden bias")
    print("observations:", observations)
    print("estimated change point:", summary["estimated_change_point"])
    print("maximum shift score:", f"{summary['max_shift_score']:.3f}")
    print("\nWindow summaries:")
    for window in summary["windows"][:4] + summary["windows"][-4:]:
        print(
            f"  {window['window_start']:>2}-{window['window_end']:>2} | "
            f"best face={window['most_likely_face']} | "
            f"TVD to fair={window['total_variation_to_fair']:.3f}"
        )


if __name__ == "__main__":
    main()
