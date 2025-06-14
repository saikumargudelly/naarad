"""Mock implementation of agent monitoring for testing."""

class MockAgentMonitor:
    def __init__(self):
        self.metrics = {}
    
    def track_metric(self, name, value, tags=None):
        self.metrics[name] = (value, tags or {})
    
    def increment_counter(self, name, value=1, tags=None):
        current = self.metrics.get(name, (0, tags or {}))[0]
        self.metrics[name] = (current + value, tags or {})
    
    def record_gauge(self, name, value, tags=None):
        self.metrics[name] = (value, tags or {})
    
    def record_histogram(self, name, value, tags=None):
        if name not in self.metrics:
            self.metrics[name] = (0, tags or {}, 0, 0, 0, 0, 0)  # count, min, max, sum, avg
        count, min_val, max_val, total, avg = self.metrics[name][1:]
        count += 1
        min_val = min(min_val, value) if count > 1 else value
        max_val = max(max_val, value) if count > 1 else value
        total += value
        avg = total / count
        self.metrics[name] = (value, tags or {}, count, min_val, max_val, total, avg)

# Create a singleton instance
agent_monitor = MockAgentMonitor()
