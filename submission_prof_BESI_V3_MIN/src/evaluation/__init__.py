"""src/evaluation — helpers de backtest et de métriques warning.

Les imports sont volontairement paresseux pour éviter de charger les
dépendances graphiques tant qu'elles ne sont pas nécessaires.
"""

__all__ = ["run_backtest", "compute_warning_metrics"]


def __getattr__(name):
	if name == "run_backtest":
		from .backtest import run_backtest

		return run_backtest
	if name == "compute_warning_metrics":
		from .warning_metrics import compute_warning_metrics

		return compute_warning_metrics
	raise AttributeError(name)
