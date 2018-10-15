values = [(1,10), (2, 9), (3, 8), (4, 7), (5, 6), (6, 5)]
scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(values)}
order = sorted(scores.keys(), key = lambda x:scores[x], reverse=True)[:3]
rollouts = [values[k] for k in order]
print(rollouts)
