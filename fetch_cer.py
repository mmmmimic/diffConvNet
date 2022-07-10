import numpy as np
with open('logs/eval_modelnet40C.log', 'r') as f:
    logs = f.read()
    content = logs.split('test acc: ')[1:]
    cer = [1 - eval(c[:5]) for c in content]

print('CER: %.4f'%np.mean(cer))