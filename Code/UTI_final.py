import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from xgboost import plot_tree

#########################################Reading Data from file and replacing strings by numbers############################################


data=pd.read_csv("pone.0194085.s001.csv")

#removal of data whose not reported values more than 50%.

#data=data.iloc[:,-4].replace(['Yes','No'],[1,0])

##############################################Data removed from original data###############################################################

dataset=data[data.columns.difference(['CVA_tenderness','abd_mass','abd_rigidity','back_pain','fatigue','vag_bleeding','vag_discharge','abd_distended2','gen_neg','pelvic_pain','ams','weakness','psychiatric_confusion','flank_pain','dec_urine_vol','diff_urinating','hematuria','polyuria','O2_Amount_First','O2_Amount_Last','O2_Amount_Max','O2_Amount_Min','O2_Amount_Mean','GCS_First','GCS_Last','ANALGESIC_AND_ANTIHISTAMINE_COMBINATION','ANALGESICS','ANESTHETICS','ANTI_OBESITY_DRUGS','ANTIALLERGY','ANTIARTHRITICS','ANTIASTHMATICS','ANTIBIOTICS','ANTICOAGULANTS','ANTIDOTES','ANTIFUNGALS','ANTIHISTAMINE_AND_DECONGESTANT_COMBINATION','ANTIHISTAMINES','ANTIHYPERGLYCEMICS','ANTIINFECTIVES','ANTIINFECTIVES_MISCELLANEOUS','ANTINEOPLASTICS','ANTIPARKINSON_DRUGS','ANTIPLATELET_DRUGS','ANTIVIRALS','AUTONOMIC_DRUGS','BIOLOGICALS','BLOOD','CARDIAC_DRUGS','CARDIOVASCULAR','CNS_DRUGS','COLONY_STIMULATING_FACTORS','CONTRACEPTIVES','COUGH_COLD_PREPARATIONS','DIAGNOSTIC','DIURETICS','EENT_PREPS','ELECT_CALORIC_H2O','GASTROINTESTINAL','HERBALS','HORMONES','IMMUNOSUPPRESANT','INVESTIGATIONAL','MISCELLANEOUS_MEDICAL_SUPPLIES__DEVICES__NON_DRUG','MUSCLE_RELAXANTS','PRE_NATAL_VITAMINS','PSYCHOTHERAPEUTIC_DRUGS','SEDATIVE_HYPNOTICS','SKIN_PREPS','SMOKING_DETERRENTS','THYROID_PREPS','UNCLASSIFIED_DRUG_PRODUCTS','VITAMINS'])]

print(dataset.shape)
dataset = dataset.replace(['yes', 'no'], [1, 0])
dataset = dataset.replace(['many','moderate','marked','few','none'], [4, 3, 2, 1, 0])
dataset = dataset.replace(['large','moderate','small','other','negative'], [ 4, 3, 2, 1, 0])
dataset = dataset.replace(['positive','other','negative'], [ 2, 1, 0])
dataset = dataset.replace(['red','orange','amber','yellow','other','colorless'], [ 5, 4, 3, 2, 1, 0])
dataset = dataset.replace(['Yes', 'No'], [1, 0])
dataset = dataset.replace(['clear', 'not_clear'], [ 1, 0])
dataset = dataset.replace('4+', 4)
dataset = dataset.replace('not_reported', 'NaN')

#full_data
dataset1=dataset[dataset.columns.difference(['PATID','ID','O2_Dependency_First','O2_Dependency_Last','abx','dispo','UTI_diag','abxUTI','race','ethnicity','lang','maritalStatus','employStatus','insurance_status','disposition','arrival','split','gender','chief_complaint','abxUTI','dispo','alt_diag'])]

#print(dataset1.shape)

########################################################Data copied to retain column name###################################################

dataset_1=dataset1

#########################################################Imputation of data#################################################################
import sklearn.preprocessing as preprocessing

imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(dataset1)
dataset_imputed = imputer.transform(dataset1)
#print(dataset_imputed.shape)
dataset1=dataset_imputed

x=np.around(dataset1)
dataset_imputed=x

dataset11 = pd.DataFrame(dataset_imputed, columns = dataset_1.columns)

####################################################Addition of Gender and chief complain###################################################

x1=pd.get_dummies(dataset[['gender','chief_complaint']])
x = pd.concat([dataset11, x1], axis=1, sort=False)
x.rename(columns = lambda x: x.replace(' ', '_'), inplace=True)
y=dataset['UTI_diag']

#print(dataset['UTI_diag'])
#print(y)

x.to_csv('dataset1.txt')
np.savetxt('result.txt',y,fmt='%1.0e')


################################################Defining, Training and Testing model##################################################################

print("\nPreparing Model\n")


#splitting the train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

ns_probs = [0 for _ in range(len(y_test))]

from xgboost import XGBClassifier

#fitting model after best parameter sets
model= XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.3,
              learning_rate=0.2, max_delta_step=0, max_depth=6,
              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=30,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=1, subsample=1, verbosity=1)

model.fit(x_train,y_train)

from sklearn.model_selection import cross_val_score
score=cross_val_score(model,x_train,y_train,cv=10)

score
score.mean()

print("\n Model is Prepared, testing is going on\n")

#predicting the test set results

y_pred=model.predict(x_test)
y_pred_all=model.predict(x_test)
np.savetxt('predicted.txt',y_pred_all,fmt='%1.0e')
np.savetxt('result_test.txt',y_test,fmt='%1.0e')


print('Accuracy of Train Data :', model.score(x_train,y_train))
print('Accuracy of Test Data :', model.score(x_test,y_test))

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#roc - auc

# predict probabilities
xgc_probs = model.predict_proba(x_test)
# keep probabilities for the positive outcome only
xgc_probs = xgc_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
xgc_auc = roc_auc_score(y_test, xgc_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('xgboost: ROC AUC=%.3f' % (xgc_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
xgc_fpr, xgc_tpr, _ = roc_curve(y_test, xgc_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(xgc_fpr, xgc_tpr, marker='.', label='xgboost')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


#######################################################SHAP Analysis to extract important features##########################################

print("\n SHAP Analysis started\n")

#shap implementation for global interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)
#shap_values = shap.TreeExplainer(model).shap_values(x)
r=shap_values
r = pd.DataFrame(shap_values, columns = x.columns)
r = r.add_suffix('_shap')

#print(r)
#print(r.shape)
r.to_csv('dataset1_shap.txt')
shap.summary_plot(shap_values, x, plot_type="bar",max_display=10)
#shap.force_plot(explainer.expected_value,shap_values[0,:],x.iloc[0,:],matplotlib=True)
#plt.show(tt)
shap.dependence_plot('ua_wbc', shap_values, x,interaction_index=None)
#shap.summary_plot(shap_values, x,max_display=10)
