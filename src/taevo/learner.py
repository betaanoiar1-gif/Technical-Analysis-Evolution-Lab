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

    def _balanced_sample(self, population: list[Strategy], target: int) -> list[Strategy]:
        """Sample an approximately balanced Long/Flat vs directional pool."""
        target = int(target)
        if target <= 0:
            return []

        shuffled = list(population)
        self.rng.shuffle(shuffled)

        directional = [s for s in shuffled if s.directional]
        long_flat = [s for s in shuffled if not s.directional]

        directional_n = target // 2
        long_flat_n = target - directional_n

        selected = []
        selected.extend(directional[:directional_n])
        selected.extend(long_flat[:long_flat_n])

        # Fill any shortfall from the remaining pool without duplicating objects.
        if len(selected) < target:
            selected_ids = {id(s) for s in selected}
            remaining = [s for s in shuffled if id(s) not in selected_ids]
            selected.extend(remaining[: target - len(selected)])

        self.rng.shuffle(selected)
        return selected[:target]

    def _balanced_elites(
        self,
        ranked: list[tuple[Strategy, dict]],
        elite_n: int,
    ) -> list[Strategy]:
        """Keep elite pressure while guaranteeing representation of both modes."""
        if not ranked:
            return []

        directional = [s for s, _ in ranked if s.directional]
        long_flat = [s for s, _ in ranked if not s.directional]

        directional_n = min(len(directional), elite_n // 2)
        long_flat_n = min(len(long_flat), elite_n - directional_n)

        # If one side cannot fill its quota, backfill from the other side.
        if directional_n < elite_n // 2:
            long_flat_n = min(
                len(long_flat),
                elite_n - directional_n,
            )

        if long_flat_n < elite_n - directional_n:
            directional_n = min(
                len(directional),
                elite_n - long_flat_n,
            )

        elites = (
            directional[:directional_n]
            + long_flat[:long_flat_n]
        )

        # Final deterministic backfill by global rank if still short.
        if len(elites) < elite_n:
            elite_ids = {id(s) for s in elites}
            for s, _ in ranked:
                if id(s) not in elite_ids:
                    elites.append(s)
                    elite_ids.add(id(s))
                if len(elites) >= elite_n:
                    break

        return elites[:elite_n]

    def _failure_adjustment(
        self,
        population: list[Strategy],
        records: list[tuple[Strategy, dict]],
    ) -> list[Strategy]:
        good_schools = [
            s.candidate.school
            for s, r in records
            if not r["rejected"]
            and r["validation"]["score"] > 0
        ]
        if not good_schools:
            return population
        best_school = max(
            set(good_schools),
            key=good_schools.count,
        )
        focused = [
            s
            for s in population
            if s.candidate.school == best_school
        ]
        return (
            focused
            + population[: max(1, len(population) // 4)]
            if len(focused) >= 5
            else population
        )

    def run(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> dict:
        cycles = int(self.cfg["search"]["cycles"])
        target = int(self.cfg["search"]["candidates_per_cycle"])
        elite_fraction = float(self.cfg["search"]["elite_fraction"])
        mutation_rate = float(self.cfg["search"]["mutation_rate"])

        population = self._initial_population()
        cycle_reports = []

        for cycle in range(cycles):
            # Keep exploration balanced before every evaluation cycle.
            population = self._balanced_sample(
                population,
                max(target * 2, target),
            )

            records: list[tuple[Strategy, dict]] = []

            for strategy in population[:target]:
                report = evaluate_development(
                    strategy,
                    train,
                    validation,
                    self.cfg,
                    self.rng,
                )
                records.append((strategy, report))
                self.history.append({
                    "cycle": cycle,
                    "strategy": strategy.name,
                    **report,
                })

            ranked = sorted(
                records,
                key=lambda x: x[1]["validation"]["score"],
                reverse=True,
            )

            elite_n = max(
                2,
                int(len(ranked) * elite_fraction),
            )

            elites = self._balanced_elites(
                ranked,
                elite_n,
            )

            adjusted = self._failure_adjustment(
                population,
                records,
            )

            half_target = target // 2
            target_directional = half_target
            target_long_flat = target - target_directional

            directional_elites = [
                s for s in elites if s.directional
            ]
            long_flat_elites = [
                s for s in elites if not s.directional
            ]

            next_directional = list(
                directional_elites[:target_directional]
            )
            next_long_flat = list(
                long_flat_elites[:target_long_flat]
            )

            # Fill both mode buckets from the elite pool. Mutation may flip
            # direction, and the child is placed into its resulting mode bucket.
            attempts = 0
            max_attempts = max(100, target * 50)

            while (
                len(next_directional) < target_directional
                or len(next_long_flat) < target_long_flat
            ) and attempts < max_attempts:
                attempts += 1

                available_parents = []
                if directional_elites:
                    available_parents.extend(directional_elites)
                if long_flat_elites:
                    available_parents.extend(long_flat_elites)

                if not available_parents:
                    break

                parent = available_parents[
                    int(
                        self.rng.integers(
                            0,
                            len(available_parents),
                        )
                    )
                ]

                child = (
                    mutate_strategy(parent, self.rng)
                    if self.rng.random() < mutation_rate
                    else parent
                )

                if child.directional:
                    if len(next_directional) < target_directional:
                        next_directional.append(child)
                else:
                    if len(next_long_flat) < target_long_flat:
                        next_long_flat.append(child)

            # Guaranteed balanced backfill from adjusted population.
            if (
                len(next_directional) < target_directional
                or len(next_long_flat) < target_long_flat
            ):
                for candidate in adjusted:
                    if (
                        len(next_directional) < target_directional
                        and candidate.directional
                    ):
                        next_directional.append(candidate)
                    elif (
                        len(next_long_flat) < target_long_flat
                        and not candidate.directional
                    ):
                        next_long_flat.append(candidate)

                    if (
                        len(next_directional) >= target_directional
                        and len(next_long_flat) >= target_long_flat
                    ):
                        break

            # Final backfill from the existing population if necessary.
            if (
                len(next_directional) < target_directional
                or len(next_long_flat) < target_long_flat
            ):
                for candidate in population:
                    if (
                        len(next_directional) < target_directional
                        and candidate.directional
                    ):
                        next_directional.append(candidate)
                    elif (
                        len(next_long_flat) < target_long_flat
                        and not candidate.directional
                    ):
                        next_long_flat.append(candidate)

                    if (
                        len(next_directional) >= target_directional
                        and len(next_long_flat) >= target_long_flat
                    ):
                        break

            population = (
                next_directional[:target_directional]
                + next_long_flat[:target_long_flat]
            )
            self.rng.shuffle(population)

            best = ranked[0]
            cycle_reports.append({
                "cycle": cycle,
                "best": best[0].name,
                "validation": best[1]["validation"],
                "rejected": best[1]["rejected"],
                "reasons": best[1]["rejection_reasons"],
            })

        final_candidates = sorted(
            self.history,
            key=lambda r: r["validation"]["score"],
            reverse=True,
        )
        accepted = [
            r
            for r in final_candidates
            if not r["rejected"]
        ]

        return {
            "cycles": cycle_reports,
            "history": self.history,
            "best_development": final_candidates[:10],
            "best_accepted_development": accepted[:10],
        }


def write_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
