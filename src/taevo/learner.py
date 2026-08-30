from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import evaluate_development
from .schools import candidate_library
from .strategies import Strategy, add_filters, mutate_strategy


class EvolutionLearner:
    """Auditable evolutionary search that is blind to the final holdout."""

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    def _initial_population(self) -> list[Strategy]:
        return add_filters(candidate_library())

    def _failure_adjustment(self, population: list[Strategy], records: list[tuple[Strategy, dict]]) -> list[Strategy]:
        good_schools = [s.candidate.school for s, r in records if not r["rejected"] and r["validation"]["score"] > 0]
        if not good_schools:
            return population
        best_school = max(set(good_schools), key=good_schools.count)
        focused = [s for s in population if s.candidate.school == best_school]
        return focused + population[: max(1, len(population) // 4)] if len(focused) >= 5 else population

    def run(self, train: pd.DataFrame, validation: pd.DataFrame) -> dict:
        cycles = int(self.cfg["search"]["cycles"])
        target = int(self.cfg["search"]["candidates_per_cycle"])
        elite_fraction = float(self.cfg["search"]["elite_fraction"])
        population = self._initial_population()
        cycle_reports = []

        for cycle in range(cycles):
            self.rng.shuffle(population)
            population = population[: max(target * 2, target)]
            records: list[tuple[Strategy, dict]] = []
            for strategy in population[:target]:
                report = evaluate_development(strategy, train, validation, self.cfg, self.rng)
                records.append((strategy, report))
                self.history.append({"cycle": cycle, "strategy": strategy.name, **report})

            ranked = sorted(records, key=lambda x: x[1]["validation"]["score"], reverse=True)
            elite_n = max(2, int(len(ranked) * elite_fraction))
            elites = [s for s, _ in ranked[:elite_n]]
            adjusted = self._failure_adjustment(population, records)
            next_population = elites.copy()
            while len(next_population) < target:
                parent = elites[int(self.rng.integers(0, len(elites)))]
                child = mutate_strategy(parent, self.rng) if self.rng.random() < self.cfg["search"]["mutation_rate"] else parent
                next_population.append(child)
            next_population.extend(adjusted[: min(8, max(0, target - len(next_population)))])
            population = next_population
            best = ranked[0]
            cycle_reports.append({"cycle": cycle, "best": best[0].name, "validation": best[1]["validation"], "rejected": best[1]["rejected"], "reasons": best[1]["rejection_reasons"]})

        final_candidates = sorted(self.history, key=lambda r: r["validation"]["score"], reverse=True)
        accepted = [r for r in final_candidates if not r["rejected"]]
        return {"cycles": cycle_reports, "history": self.history, "best_development": final_candidates[:10], "best_accepted_development": accepted[:10]}


def write_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
