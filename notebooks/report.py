from typing import List
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve,f1_score, accuracy_score, balanced_accuracy_score, brier_score_loss
import sys
sys.path.append("../")
# from diabnet.model import load
from diabnet.apply import Predictor
from diabnet.data import get_feature_names

class Dataset:
    def __init__(self, fn: str, feat_names: List[str], predictor: Predictor):
        self.df = pd.read_csv(fn)
        self._feat_names = feat_names
        self._features = None
        self._labels = None
        self._predictor = predictor
        self._predictions = None
        self._predictions_per_age = None
        self._predictions_years_before = None
         
    @property
    def features(self):
        if self._features is None:
            self._features = self.df[self._feat_names].values
        return self._features

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.df['T2D'].values
        return self._labels

    @property
    def predictions(self):
        if self._predictions is None:
            self._predictions = np.array([np.median(self._predictor.patient(f, age=-1, samples=100)) for f in self.features])
        return self._predictions

    @property
    def predictions_per_age(self):
        if self._predictions_per_age is None:
            self._predictions_per_age = {age:np.array([np.median(self._predictor.patient(f, age=age, samples=100)) for f in self.features]) for age in range(20,80,5)}
        return self._predictions_per_age

    @property
    def predictions_years_before(self):
        # Probably this analysis make sense only in patients with known first diagnosis 
        i = self._feat_names.index('AGE')
        if self._predictions is None:
            self._predictions_years_before = {yb:np.array([np.median(self._predictor.patient(f, age=f[i]-yb, samples=30)) for f in self.features]) for yb in range(0,10,1)}
        return self._predictions_years_before

#################################

class DiabNetReport:
    def __init__(self, nn_pth, data_suffix, use_negatives=True, use_bmi=False, use_sex=True, use_parents=True):
        # data_suffix -> e.g. "positivo_1000_random_0.csv"
        self.data_suffix = data_suffix

        # self.feat_names = self.get_feature_names("../datasets/visits_sp_unique_test_"+data_suffix, use_bmi, use_sex, use_parents)
        self.feat_names = get_feature_names("../datasets/visits_sp_unique_test_"+data_suffix, BMI=use_bmi, sex=use_sex, parents_diagnostics=use_parents)
        if use_negatives:
            negatives_csv="../datasets/visits_sp_unique_test_"+data_suffix.strip(".csv")+"_negatives_older60.csv"
        else:
            negatives_csv=None
        self.predictor = Predictor(nn_pth, self.feat_names, negatives_csv=negatives_csv)
        self._dataset_test_unique = None
        self._dataset_test_unique_subset_older50 = None
        self._dataset_test_changed = None
        self._dataset_test_all = None
        self._dataset_test_first_diag = None
        self._negatives_older60 = None



    @property
    def negatives_older60(self):
        if self._negatives_older60 is None:
            self._negatives_older60 = self.predictor.negatives_life(age_range=range(20,80,5), bmi=-1, samples=100)
        return self._negatives_older60

    @property
    def dataset_test_unique(self):
        if self._dataset_test_unique is None:
            self._dataset_test_unique = Dataset("../datasets/visits_sp_unique_test_"+self.data_suffix, self.feat_names, self.predictor)
        return self._dataset_test_unique


    @property
    def dataset_test_unique_subset_older50(self):
        # subset positives or negatives older than 50
        if self._dataset_test_unique_subset_older50 is None:
            db = Dataset("../datasets/visits_sp_unique_test_"+self.data_suffix, self.feat_names, self.predictor)
            db.df = db.df[(db.df.T2D > 0.5) | (db.df.AGE >= 50)]
            self._dataset_test_unique_subset_older50 = db
        return self._dataset_test_unique_subset_older50
    
    @property
    def dataset_test_changed(self):
        if self._dataset_test_changed is None:
            self._dataset_test_changed = Dataset("../datasets/visits_sp_changed_test_"+self.data_suffix, self.feat_names, self.predictor)
        return self._dataset_test_changed

    @property
    def dataset_test_first_diag(self):
        if self._dataset_test_first_diag == None:
            db = Dataset("../datasets/visits_sp_changed_test_"+self.data_suffix, self.feat_names, self.predictor)
            # seleciona as amostras negativas e agrupa por id pegando a MAIOR idade
            neg  = db.df[db.df.T2D == 0][['id','AGE']].groupby('id').max()
            # seleciona as amostras positivas e agrupa por id pegando a MENOR idade
            pos  = db.df[db.df.T2D == 1][['id','AGE']].groupby('id').min()
            dtmp = neg.join(pos, lsuffix='_neg', rsuffix='_pos')
            # filtra somente pacientes que a idade negativa < idade positiva
            dtmp = dtmp[dtmp.AGE_neg < dtmp.AGE_pos]
            tmp = pd.concat([pd.DataFrame({'AGE':dtmp.AGE_neg}),pd.DataFrame({'AGE':dtmp.AGE_pos})])
            tmp = tmp.reset_index()
            # Eu odeio a API do pandas!!!
            db.df = db.df.merge(tmp, on=['id','AGE'], how="inner")
            self._dataset_test_first_diag = db
        return self._dataset_test_first_diag

    @property
    def dataset_test_all(self):
        if self._dataset_test_all is None:
            self._dataset_test_all = Dataset("../datasets/visits_sp_all_test_"+self.data_suffix, self.feat_names, self.predictor)
        return self._dataset_test_all

######################################################
    # Report methods
    def auc(self):
        db = self.dataset_test_unique
        auc = roc_auc_score(db.labels, db.predictions)
        print("AUC = {:.4f}".format(auc)) 

    def f1(self):
        db = self.dataset_test_unique
        f1 = f1_score(db.labels, np.round(db.predictions))
        print("F1-Score = {:.4f}".format(f1)) 

    def acc(self):
        db = self.dataset_test_unique
        acc = accuracy_score(db.labels, np.round(db.predictions))
        print("Accuracy = {:.4f}".format(acc)) 

    def bacc(self):
        db = self.dataset_test_unique
        bacc = balanced_accuracy_score(db.labels, np.round(db.predictions))
        print("Balanced Accuracy = {:.4f}".format(bacc)) 

    def auc_per_age(self):
        db = self.dataset_test_unique
        aucs = {age:roc_auc_score(db.labels, preds) for age, preds in db.predictions_per_age.items()}
        for k, v in aucs.items():
            print("Predictions using patient age = {} have AUC = {:.4f}".format(k,v))  

    def f1_per_age(self):
        db = self.dataset_test_unique
        f1s = {age:f1_score(db.labels, np.round(preds)) for age, preds in db.predictions_per_age.items()}
        for k, v in f1s.items():
            print("Predictions using patient age = {} have F1-Score = {:.4f}".format(k,v)) 


    def acc_per_age(self):
        db = self.dataset_test_unique
        accs = {age:accuracy_score(db.labels, np.round(preds)) for age, preds in db.predictions_per_age.items()}
        for k, v in accs.items():
            print("Predictions using patient age = {} have Accuracy = {:.4f}".format(k,v)) 

    def bacc_per_age(self):
        db = self.dataset_test_unique
        baccs = {age:balanced_accuracy_score(db.labels, np.round(preds)) for age, preds in db.predictions_per_age.items()}
        for k, v in baccs.items():
            print("Predictions using patient age = {} have Balanced Accuracy = {:.4f}".format(k,v)) 

    def roc_curves(self):
        db = self.dataset_test_unique
        for _, pr in db.predictions_per_age.items():
            fpr, tpr, threshold = roc_curve(db.labels, pr)
            plt.plot(fpr, tpr)
        plt.ylim(0,1)
        plt.xlim(0,1)
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    def precision_recall_curves(self):
        db = self.dataset_test_unique
        for _, pr in db.predictions_per_age.items():
            precision, recall, _ = precision_recall_curve(db.labels, pr)
            plt.step(recall, precision,where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.show()

    # Reports for subset older 50
    def auc_older50(self):
        db = self.dataset_test_unique_subset_older50
        auc = roc_auc_score(db.labels, db.predictions)
        print("AUC = {:.4f}".format(auc)) 

    def f1_older50(self):
        db = self.dataset_test_unique_subset_older50
        f1 = f1_score(db.labels, np.round(db.predictions))
        print("F1-Score = {:.4f}".format(f1)) 

    def acc_older50(self):
        db = self.dataset_test_unique_subset_older50
        acc = accuracy_score(db.labels, np.round(db.predictions))
        print("Accuracy = {:.4f}".format(acc)) 

    def bacc_older50(self):
        db = self.dataset_test_unique_subset_older50
        bacc = balanced_accuracy_score(db.labels, np.round(db.predictions))
        print("Balanced Accuracy = {:.4f}".format(bacc)) 

    def auc_per_age_older50(self):
        db = self.dataset_test_unique_subset_older50
        aucs = {age:roc_auc_score(db.labels, preds) for age, preds in db.predictions_per_age.items()}
        for k, v in aucs.items():
            print("Predictions using patient age = {} have AUC = {:.4f}".format(k,v))  

    def f1_per_age_older50(self):
        db = self.dataset_test_unique_subset_older50
        f1s = {age:f1_score(db.labels, np.round(preds)) for age, preds in db.predictions_per_age.items()}
        for k, v in f1s.items():
            print("Predictions using patient age = {} have F1-Score = {:.4f}".format(k,v)) 

    def acc_per_age_older50(self):
        db = self.dataset_test_unique_subset_older50
        accs = {age:accuracy_score(db.labels, np.round(preds)) for age, preds in db.predictions_per_age.items()}
        for k, v in accs.items():
            print("Predictions using patient age = {} have Accuracy = {:.4f}".format(k,v)) 

    def bacc_per_age_older50(self):
        db = self.dataset_test_unique_subset_older50
        baccs = {age:balanced_accuracy_score(db.labels, np.round(preds)) for age, preds in db.predictions_per_age.items()}
        for k, v in baccs.items():
            print("Predictions using patient age = {} have Balanced Accuracy = {:.4f}".format(k,v)) 

    def roc_curves_older50(self):
        db = self.dataset_test_unique_subset_older50
        for _, pr in db.predictions_per_age.items():
            fpr, tpr, threshold = roc_curve(db.labels, pr)
            plt.plot(fpr, tpr)
        plt.ylim(0,1)
        plt.xlim(0,1)
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    def precision_recall_curves_older50(self):
        db = self.dataset_test_unique_subset_older50
        for _, pr in db.predictions_per_age.items():
            precision, recall, _ = precision_recall_curve(db.labels, pr)
            plt.step(recall, precision,where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.show()
    
    # family test
    def family_plots(self):
        db = self.dataset_test_unique
        df_pedigree = pd.read_csv("../datasets/pedigree.csv")
        df_pedigree = df_pedigree.set_index('id')

        ids = db.df[['id', 'AGE', 'sex', 'mo', 'fa', 'mo_t2d', 'fa_t2d', 'T2D']]
        ids = ids.join(df_pedigree[['famid']], how='left', on='id')

        color_by_family = ids['famid'].values
        famid_sorted = np.unique(np.sort(color_by_family))
        fam_masks = [color_by_family == i for i in famid_sorted]
        patient_ages = np.array(db.features[:,db._feat_names.index('AGE')])
        # print(patient_ages)


        plt.figure(figsize=(16,32))
        for i in range(len(fam_masks)):
            ax1=plt.subplot(8, 4, i+1)
            p = np.stack([db.predictions_per_age[age][fam_masks[i]] for age in range(20,80,5)], axis=0)
            m_p = db.labels[fam_masks[i]] == 1
            m_n = db.labels[fam_masks[i]] == 0

            pa = patient_ages[fam_masks[i]]
            pp = db.predictions[fam_masks[i]]

            if True in m_n:
                plt.plot(p[:,m_n], color='grey')
            if True in m_p:
                plt.plot(p[:,m_p], color='red')
            # print patient age 
            plt.scatter((pa[m_p]-20)/5, pp[m_p], c='black', marker='^')
            plt.scatter((pa[m_n]-20)/5, pp[m_n], c='blue', marker='+')

            plt.title("famid {}".format(famid_sorted[i]))
            plt.xticks(range(12), np.arange(20,80,5))
            plt.ylim(0,1)
            # break
        plt.show()

    def parents_plots(self):
        db = self.dataset_test_unique
        df = db.df.set_index('id')
        # print(df)
        df_fathers = df[df.index.isin(df.fa)]
        df_mothers = df[df.index.isin(df.mo)]
        df_children = df[(df.fa.isin(df_fathers.index)) & (df.mo.isin(df_mothers.index))]
        # print(df_children)
        parents = df_children[['fa','mo']].groupby(['fa','mo']).count().index
        predictions = np.stack([db.predictions_per_age[age] for age in range(20,80,5)], axis=1)
        patient_ages = np.array(db.features[:,db._feat_names.index('AGE')])
        # print(patient_ages)


        plt.figure(figsize=(16,36))
        for i in range(len(parents)):
            ax1=plt.subplot(9, 4, i+1)
            plt.title("father {} mother {}".format(parents[i][0], parents[i][1]))
            father_mask = (db.df['id'] == parents[i][0]).values
            mother_mask = (db.df['id'] == parents[i][1]).values
            pos_mask = db.labels == 1
            neg_mask = db.labels == 0
            children_mask = np.logical_and((db.df['fa'] == parents[i][0]).values, (db.df['mo'] == parents[i][1]).values)

            m_pos_fa = np.logical_and(pos_mask, father_mask)
            m_pos_mo = np.logical_and(pos_mask, mother_mask)
            m_pos_chi = np.logical_and(pos_mask, children_mask)
            m_neg_fa = np.logical_and(neg_mask, father_mask)
            m_neg_mo = np.logical_and(neg_mask, mother_mask)
            m_neg_chi = np.logical_and(neg_mask, children_mask)

            if True in m_pos_fa:
                plt.plot(np.transpose(predictions[m_pos_fa,:]), color='blue')
            if True in m_pos_mo:
                plt.plot(np.transpose(predictions[m_pos_mo,:]), color='red')
            if True in m_pos_chi:
                plt.plot(np.transpose(predictions[m_pos_chi,:]), color='green')
            if True in m_neg_fa:
                plt.plot(np.transpose(predictions[m_neg_fa,:]), color='blue', ls='--')
            if True in m_neg_mo:
                plt.plot(np.transpose(predictions[m_neg_mo,:]), color='red', ls='--')
            if True in m_neg_chi:
                plt.plot(np.transpose(predictions[m_neg_chi,:]), color='green', ls='--');
            plt.scatter((patient_ages[father_mask]-20)/5, db.predictions[father_mask], c='blue')
            plt.scatter((patient_ages[mother_mask]-20)/5, db.predictions[mother_mask], c='red')
            plt.scatter((patient_ages[children_mask]-20)/5, db.predictions[children_mask], c='green')
            plt.ylim(0,1)
            plt.xticks(range(12), np.arange(20,80,5))
        plt.show()

    # def first_positives(self):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
    #     age_above_50 = db.df.AGE.values[pos_mask] >= 50
    #     age_below_50 = db.df.AGE.values[pos_mask] < 50

    #     neg = self.negatives_older60
    #     pred_below_50 = [np.array(x) for x in pred[:,age_below_50]]
    #     pred_above_50 = [np.array(x) for x in pred[:,age_above_50]]

    #     plt.figure(figsize=(15,5))
    #     plt.subplot(131)
    #     sns.boxplot(x=[i for i in neg[1]], y=pred_below_50, showfliers=False)
    #     plt.xlabel("positives\n(age < 50 )")
    #     plt.ylim(0,1)
    #     plt.subplot(132)
    #     sns.boxplot(x=[i for i in neg[1]], y=pred_above_50, showfliers=False)
    #     plt.xlabel("positives\n(age >= 50 )")
    #     plt.ylim(0,1)
    #     plt.subplot(133)
    #     sns.boxplot(x=[i for i in neg[1]], y=neg[0], showfliers=False)
    #     plt.xlabel("negatives\n(age >= 60)")
    #     plt.ylim(0,1)

    #     plt.show()

    def first_positives(self):
        db = self.dataset_test_first_diag
        pos_mask = db.labels == 1
        pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
        age_above_50 = db.df.AGE.values[pos_mask] >= 50
        age_below_50 = db.df.AGE.values[pos_mask] < 50

        neg = self.negatives_older60
        pred_below_50 = [np.array(x) for x in pred[:,age_below_50]]
        pred_above_50 = [np.array(x) for x in pred[:,age_above_50]]

        color_boxplot = sns.color_palette("cool", n_colors=20)

        plt.figure(figsize=(15,5))
        plt.subplot(131)
        sns.boxplot(x=[i for i in neg[1]], y=pred_below_50, showfliers=False, palette=[color_boxplot[int(np.median(a)*20)] for a in pred_below_50])
        plt.xlabel("positives\n(age < 50 )")
        plt.ylim(0,1)
        plt.subplot(132)
        sns.boxplot(x=[i for i in neg[1]], y=pred_above_50, showfliers=False, palette=[color_boxplot[int(np.median(a)*20)] for a in pred_above_50])
        plt.xlabel("positives\n(age >= 50 )")
        plt.ylim(0,1)
        plt.subplot(133)
        sns.boxplot(x=[i for i in neg[1]], y=neg[0], showfliers=False, palette=[color_boxplot[int(np.median(a)*20)] for a in neg[0]])
        plt.xlabel("negatives\n(age >= 60)")
        plt.ylim(0,1)

        plt.show()


    # def use_first_diag(self):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     neg_mask = db.labels == 0
    #     print(db.predictions[neg_mask],db.predictions[pos_mask])
    #     print((db.predictions[pos_mask]-db.predictions[neg_mask])*100)
    #     print(np.mean((db.predictions[pos_mask]-db.predictions[neg_mask]))*100)



    # def years_before_diag(self, age_at_prediction=20):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     pred = db.predictions_per_age[age_at_prediction][pos_mask]
    #     age = db.df.AGE.values[pos_mask]-age_at_prediction
    #     # print(pred)
    #     # print(age)
    #     plt.scatter(pred,age)
    #     coef = np.polyfit(pred,age,1)
    #     poly1d_fn = np.poly1d(coef)
    #     print(coef)
    #     plt.plot(pred, age, 'ko', pred, poly1d_fn(pred), 'r')
    #     plt.xlim(0,1)
    #     plt.xlabel("Predicted probability (30yo)")
    #     plt.ylabel("Patient age at first positive diagnostic")
    #     plt.show()

    # def years_before_diag2(self):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
    #     age_above_50 = db.df.AGE.values[pos_mask] > 50
    #     age_below_50 = db.df.AGE.values[pos_mask] < 50

    #     pred_above50_mean = np.mean(pred[:,age_above_50], axis=1)
    #     pred_above50_std = np.mean(pred[:,age_above_50], axis=1)
    #     pred_below50_mean = np.mean(pred[:,age_below_50], axis=1)
    #     pred_below50_std = np.std(pred[:,age_below_50], axis=1)

    #     neg = self.negatives_older60
    #     neg_mean = np.mean(np.array(neg[0]), axis=1)
    #     # neg_median = np.median(np.array(neg[0]), axis=1)
    #     neg_std = np.std(np.array(neg[0]), axis=1)

    #     # # age_inter =  np.logical_and((db.df.AGE.values[pos_mask] > 30), (db.df.AGE.values[pos_mask] < 70))
    #     # plt.plot(pred_above50_mean, color='green')
    #     plt.errorbar(x=np.arange(20.2,80.2,5), y=pred_above50_mean, yerr=pred_above50_std, color='green')
    #     # # plt.plot(np.mean(pred[:,age_inter], axis=1), color='yellow')
    #     # plt.plot(pred_below50_mean, color='red')
    #     plt.errorbar(x=np.arange(20.1,80.1,5), y=pred_below50_mean, yerr=pred_below50_std, color='red')
    #     # plt.plot(neg_mean, color='grey')
    #     plt.errorbar(x=np.arange(20,80,5), y=neg_mean, yerr=neg_std, color='grey')
    #     # plt.plot(neg_median, color='purple')
    #     # # plt.scatter(age, db.predictions[pos_mask])
    #     plt.show()



    

if __name__ == "__main__":
    report = DiabNetReport("../diabnet/models/teste-sp-soft-label_tmp.pth", "positivo_1000_random_0.csv", "../datasets/visits_sp_unique_test_positivo_1000_random_0_negatives_older60.csv")
    report.first_positives2()
    # report.auc_per_age()
    # report.f1_per_age()
    # report.acc_per_age()
    # report.bacc_per_age()
    # report.roc_curves()


    # report.precision_recall_curves()
    # # subset older 60
    # report.auc_per_age_older50()
    # report.f1_per_age_older50()
    # report.acc_per_age_older50()
    # report.bacc_per_age_older50()
    # report.roc_curves_older50()
    # report.precision_recall_curves_older50()

    # print(report.dataset_test_changed.df)
    # report.family_plots()
    # report.parents_plots()
    # report.family_plots()


    # print(report.feat_names)
    # print(report.dataset_test_unique.features)
    # print(report.dataset_test_unique.labels)
    # print(report.dataset_test_unique.predictions)