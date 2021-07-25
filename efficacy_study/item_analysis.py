import csv
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

'''
by Raefah Wahid
'''

def experience_analysis(data, item):
    low = []
    medium = []
    high = []
    row = 0
    for elt in data['Experience?']:
        if elt == 0:
            low.append(data[item][row])
        elif elt == 1:
            medium.append(data[item][row])
        else:
            high.append(data[item][row])
        row += 1

    plt.scatter(['Low']*(len(low)), low, c='cyan')
    plt.scatter(['Medium']*(len(medium)), medium, c='magenta')
    plt.scatter(['High']*(len(high)), high, c='green')
    plt.show()

    # one-way ANOVA with experience against quiz1
    anova = stats.f_oneway(low, medium, high)
    return anova

def language_analysis(data, item):
    english = []
    turkish = []
    bilingual = []
    chinese = []
    spanish = []
    portuguese = []
    tamil = []
    row = 0
    for elt in data['Native_Lang']:
        if elt == "English":
            english.append(data[item][row])
        elif elt == "Turkish":
            turkish.append(data[item][row])
        elif elt == "Chinese":
            chinese.append(data[item][row])
        elif elt == "Spanish":
            spanish.append(data[item][row])
        elif elt == "Portuguese":
            portuguese.append(data[item][row])
        elif elt == "Tamil":
            tamil.append(data[item][row])
        else:
            bilingual.append(data[item][row])
        row += 1

    non_english = turkish + chinese + spanish + portuguese + tamil
    # one-way ANOVA with native language against s2
    anova = stats.f_oneway(english, non_english, bilingual)

    plt.scatter(['English']*(len(english)), english, c='cyan')
    plt.scatter(['Non-English']*(len(non_english)), non_english, c='magenta')
    plt.scatter(['Bilingual']*(len(bilingual)), bilingual, c='green')
    plt.show()

    return anova


if __name__ == '__main__':
    data = pd.read_csv('emods_data.csv')

    ex_q1_i1 = experience_analysis(data, "Q1_I1")
    print(ex_q1_i1)
    ex_q1_i2 = experience_analysis(data, "Q1_I2")
    print(ex_q1_i2)
    # ex_q1_i3 = experience_analysis(data, "Q1_I3") # everyone did well on this, excluding
    # print(ex_q1_i3)
    ex_q1_i4 = experience_analysis(data, "Q1_I4")
    print(ex_q1_i4)

    ex_q2_i1 = experience_analysis(data, "Q2_I1")
    print(ex_q2_i1)
    ex_q2_i2 = experience_analysis(data, "Q2_I2")
    print(ex_q2_i2)
    ex_q2_i3 = experience_analysis(data, "Q2_I3")
    print(ex_q2_i3)
    ex_q2_i4 = experience_analysis(data, "Q2_I4")
    print(ex_q2_i4)

    nl_q1_i1 = language_analysis(data, "Q1_I1")
    print(nl_q1_i1)
    nl_q1_i2 = language_analysis(data, "Q1_I2")
    print(nl_q1_i2)
    # nl_q1_i3 = language_analysis(data, "Q1_I3") # everyone did well on this, excluding
    # print(nl_q1_i3)
    nl_q1_i4 = language_analysis(data, "Q1_I4")
    print(nl_q1_i4)

    nl_q2_i1 = language_analysis(data, "Q2_I1")
    print(nl_q2_i1)
    nl_q2_i2 = language_analysis(data, "Q2_I2")
    print(nl_q2_i2)
    nl_q2_i3 = language_analysis(data, "Q2_I3")
    print(nl_q2_i3)
    nl_q2_i4 = language_analysis(data, "Q2_I4")
    print(nl_q2_i4)
