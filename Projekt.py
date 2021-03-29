import snorkel

import glob
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import labeling_function
from Datencorrelation import load_spam_dataset
###--------------------------------------

df_train, df_test, df = load_spam_dataset()
#Testlabel zum vergleich
Y_test = df_test.label.values
###--------------------------------------
# For clarity, we define constants to represent the class labels
ABSTAIN = -1
NORAIN = 0
RAIN = 1
###--------------------------------------


@labeling_function()
def rainfall(x):
    if x.rainfall >= 0 and x.rainfall <= 0.8:
        return ABSTAIN
    elif x.rainfall > 0.8:
        return RAIN

@labeling_function()
def humidity3pm(x):
    if  x.humidity3pm >= 46 and x.humidity3pm <= 68:
        return ABSTAIN
    elif x.humidity3pm > 68:
        return RAIN
    else:
        return NORAIN
    
@labeling_function()
def humidity9am(x):
    if  x.humidity9am >= 64 and x.humidity9am <= 77:
        return ABSTAIN
    elif x.humidity9am > 77:
        return RAIN
    else:
        return NORAIN

@labeling_function()
def cloud3pm(x):
    if x.cloud3pm >= 3 and x.cloud3pm <= 7:
        return ABSTAIN
    elif x.cloud3pm > 7:
        return RAIN
    else:
        return NORAIN

@labeling_function()
def windGustSpeed(x):
    if x.windgustspeed >= 37 and x.windgustspeed <= 44:
        return ABSTAIN
    elif x.windgustspeed > 44:
        return RAIN
    else:
        return NORAIN

@labeling_function()
def sunshine(x):
    if x.sunshine >= 4 and x.sunshine <= 9:
        return ABSTAIN
    elif x.sunshine < 4:
        return RAIN
    else:
        return NORAIN

###--------------------------------------
lfs = [
    rainfall,
    humidity3pm,
    humidity9am,
    cloud3pm,
    windGustSpeed,
    sunshine
]
###--------------------------------------
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)
###---------------------------------------
coverage_check_out, coverage_check = (L_train != ABSTAIN).mean(axis=0)
print(f"check_out coverage: {coverage_check_out * 100:.1f}%")
print(f"check coverage: {coverage_check * 100:.1f}%")

###---------------------------------------
#Zwischenstatus
from snorkel.labeling import LFAnalysis
LFAnalysis(L=L_train, lfs=lfs).lf_summary()
###---------------------------------------
from snorkel.labeling.model import MajorityLabelVoter
#Majoritiy Voter
majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)

###---------------------------------------
majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")
