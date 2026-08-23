from monitor.performance_monitor import PerformanceMonitor


class _Stats:
    @staticmethod
    def get_stats():
        return {}


def test_threshold_alert_is_deduplicated_until_metric_recovers():
    monitor = PerformanceMonitor(_Stats(), _Stats())

    monitor._check_threshold("agent_success_rate", 0.5, "agent-a")
    monitor._check_threshold("agent_success_rate", 0.4, "agent-a")

    assert len(monitor._alerts) == 1
    assert len(monitor._active_alerts) == 1

    monitor._check_threshold("agent_success_rate", 1.0, "agent-a")
    assert not monitor._active_alerts
    assert monitor._alerts[0].resolved is True

    monitor._check_threshold("agent_success_rate", 0.5, "agent-a")
    assert len(monitor._alerts) == 2
