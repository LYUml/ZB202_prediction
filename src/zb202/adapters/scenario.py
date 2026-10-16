"""Read explicit assumed occupancy; presence sensors are not people counts."""


class ScenarioInputs:
    def __init__(self, schedule):
        if not schedule or schedule[0]['minute'] != 0:
            raise ValueError('Occupancy schedule must start at minute 0')
        previous = -1
        for item in schedule:
            if item['minute'] <= previous or item['occupants'] < 0:
                raise ValueError('Schedule must be ordered, unique and nonnegative')
            previous = item['minute']
        self.schedule = schedule

    def occupants_at(self, elapsed_seconds):
        occupants = self.schedule[0]['occupants']
        for item in self.schedule:
            if item['minute'] * 60 > elapsed_seconds:
                break
            occupants = item['occupants']
        return occupants
