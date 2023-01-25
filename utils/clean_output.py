# %%
import numpy as np
def clean_epi(data):

    y = data['epi']
    epi_self = np.arange(1, 6, 1)
    epi_parents = np.arange(6, 11, 1)
    epi_teacher = np.arange(11, 16, 1)
    epi_reatt = np.array([16, 17, 18, 19, 21])
    epi_cause = np.array([22, 23, 24, 25, 26, 27, 20])
    epi_non = np.array([34, 228, 229, 230, 231, 232, 233, 234, 235, 236, 33])
    epi_inf = np.array([30, 224, 32, 31, 98])
    epi_missing = np.array([28, 29, 222, 227, 98])
    epi_ignore = np.array([0, 34, 30, -1])

    for i in epi_self:
        y[y == i] = '1'

    for i in epi_parents:
        y[y == i] = '2'

    for i in epi_teacher:
        y[y == i] = '3'

    for i in epi_reatt:
        y[y == i] = '4'

    for i in epi_cause:
        y[y == i] = '5'

    for i in epi_non:
        y[y == i] = '-1'

    for i in epi_inf:
        y[y == i] = '-1'

    for i in epi_missing:
        y[y == i] = '-1'

    for i in epi_ignore:
        y[y == i] = '-1'

    return y

def clean_soc(data):
    y = data['soc']

    soc_Externalization = np.array([0, 201, 219])
    soc_Acceptence = np.array([204, 205, 206, 207])
    soc_Elicitation = np.array([222, 223, 224])
    soc_Conflict = np.array([208, 209, 210, 211, 212, 213, 214, 215])
    soc_Integration = np.array([22, 221])

    for i in soc_Externalization:
        y[y == i] = 35

    for i in soc_Acceptence:
        y[y == i] = 39

    for i in soc_Elicitation:
        y[y == i] = 36

    for i in soc_Conflict:
        y[y == i] = 37

    for i in soc_Integration:
        y[y == i] = 41


    return y


