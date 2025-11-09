    def get_reward(target, collision, action, min_laser):
        if target:
            return 120.0
        elif collision:
            return -120.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - r3(min_laser)-0.1