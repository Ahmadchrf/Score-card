from librairies_score import *

# Cette fonction nous donne une nouvelle base de données appliquant le label encoding aux variables qualitatives encore sous le format string/object

def label_encoding(data,target):
    """
    * data  : La base de données ;
    * target : La variable cible.
    """
    
    var_disc = data.select_dtypes(include=[np.object]).columns
    
    for j in var_disc :
        
        l = list(data[j].unique())
    
        list_prop = []
        for i in l :
            list_prop.append(data[data[j] == i][target].sum()/data[data[j] == i].shape[0])
            
        dict_label = dict(zip(l,list_prop))
        lab = sorted(dict_label,key=dict_label.get,reverse = True)
        dict_fin = dict(zip(lab,reversed(range(len(lab)))))
        
        df = data.replace({j: dict_fin})
        
    return df

# Cette fonction permet de fournir selon la discrétisation choisit le graphique de stabilité par rapport à la target dans le temps
   
def stabilty_def(data,var_disc, var_time,var_count,target,t):
    
        """ * data : La base de données ;
        * var_disc :  la variable qui vient d'être discrétiser ;
        * var_time : la variable représentant la temporalité de notre base ;
        * var_count : une variable unique attribuée à chaque ligne de notre base (typiquement un id_contrat ou même l'index de la base) ;
        * taregt : la variable cible ;
        t : la fréquence temporelle.
    """
        
    name_var = list(pd.DataFrame(data[var_disc].value_counts()).index)
    name_var = sorted(name_var)

    list_fin = []
    list_fin_2 = []
    for i in range(len(name_var)) :

        df_base = data[data[var_disc]==name_var[i]].groupby([var_time]).agg({var_count:"count"})

        df_1 = data[(data[var_disc]== name_var[i]) & (data[target]==1)] 

        df_2 = df_1.groupby(data[var_time]).agg({var_time:"count"})

        df_3 = df_base.merge(df_2,left_on  = df_base.index,right_on = df_2.index,how = "left").fillna(0)

        df_3 = df_3.set_index("key_0")
        #df_3 = df_3.resample("{}M".format(t)).sum()

        df_3 = df_3.iloc[1:,:]

        try :
            if df_3.index[-1] > max(df_2.index) :
                df_3 = df_3.iloc[:-1,:]
        except :
            pass

        list_fin.append(np.round((df_3[var_time]/df_3[var_count])*100,3))

        nb = len(list_fin[0])
        def fct_temp(x):
            list_fin_2 = []
            a = 0
            for i in range(round(nb/t)):
                list_fin_2.append(round(np.mean(list_fin[x][a:a+t]),2))
                a = a + t

            return list_fin_2

    list_fin_3 = list(map(fct_temp,list(range(len(name_var)))))

    c = -1
    for impacts in list_fin_3:
        
    # calcul du taux moyen par modalités :
        c = c + 1
        
        n = data[(data[var_disc]== name_var[c]) & (data[target] == 1)].shape[0]
        m = data[(data[var_disc]== c)].shape[0]
        tx_moy = np.round(((n/m)*100),2)
    
        d = plt.plot(impacts,label = 'modalité {} : {} %'.format(name_var[c],tx_moy))
        d = plt.xlabel('dates')
        d = plt.ylabel('défaut en %')
    
    plt.title("Evolution du taux de défaut par modalité de {}".format(var_disc))
    plt.legend()
    plt.figure(figsize = (20,20))
    plt.show()
    return

# Cette fonction permet de fournir selon la discrétisation choisit le graphique de stabilité en répartition dans le temps

def stabilty_repartition(data,var_disc,var_time,var_count,t):
    
        """ * data : La base de données ;
        * var_disc :  la variable qui vient d'être discrétiser ;
        * var_time : la variable représentant la temporalité de notre base ;
        * var_count : une variable unique attribuée à chaque ligne de notre base (typiquement un id_contrat ou même l'index de la base) ;
        t : la fréquence temporelle.
    """

    df_base = data.groupby([var_time]).agg({var_count:"count"})
    
    name_var = list(pd.DataFrame(data[var_disc].value_counts()).index)
    name_var = sorted(name_var)

    list_fin = []

    for i in range(len(name_var)) : 

        df_1 = data[(data[var_disc]== name_var[i])] 

        df_2 = df_1.groupby(data[var_time]).agg({var_time:"count"})

        df_3 = df_base.merge(df_2,left_on  = df_base.index,right_on = df_2.index,how = "left").fillna(0)
        
        df_3 = df_3.set_index("key_0")
        #df_3 = df_3.resample("{}M".format(t)).sum()
        
        df_3 = df_3.iloc[1:,:]
        
        if df_3.index[-1] > max(df_2.index) :
            df_3 = df_3.iloc[:-1,:]
        else :
            pass
        
                
        list_fin.append(np.round(df_3[var_time] / df_3[var_count],5)*100)
        
    nb = len(list_fin[0])
    def fct_temp(x):
        list_fin_2 = []
        a = 0
        for i in range(round(nb/t)):
            list_fin_2.append(round(np.mean(list_fin[x][a:a+t]),2))
            a = a + t

        return list_fin_2

    list_fin_3 = list(map(fct_temp,list(range(len(name_var)))))
        
    a = -1
    for impacts in list_fin_3:
        a = a + 1
        d = plt.plot(impacts,label = 'modalité %s'%name_var[a])
        d = plt.xlabel('dates')
        d = plt.ylabel('répartition en %')
    
    plt.title("Evolution de la répartition par modalité")
    plt.legend()
    #plt.figure(figsize = (20,20))
    plt.show()
    
    return 

# Cette fonction permet de rendre compte de la méthgode des vingtiles permettant de discrétiser nos variables quantitatives à l'aide d'un tableau facilitant l'opération ainsi que d'un graphique résumant le tableau détaillé
    
def find_bin(data,var_names,target):

    """ * data : la base de données ;
        * var_names : le nom de la variable que l'on cherche à discrétiser ;
        * target : la variable cible.
    """
    a = 0
    liste = []
    for i in range(19):
        a = a + 0.05
        liste.append(np.round(a,2))

    df_1 = pd.DataFrame(data[var_names].describe(percentiles = liste)).iloc[3:,]

    liste_count = []
    for i in range(df_1.shape[0]-1):
        liste_count.append(data[(data[var_names] >= df_1.iloc[i,0]) & \
                                     (data[var_names] <= df_1.iloc[i+1,0])].shape[0])

    liste_count_def = []
    for i in range(df_1.shape[0]-1):
        liste_count_def.append(data[data[target]==1][(data[var_names] >= df_1.iloc[i,0]) & \
                                     (data[var_names] <= df_1.iloc[i+1,0])].shape[0])

    data_fin = pd.DataFrame([liste_count,liste_count_def],index = ["nombre","defaut"]).T
    data_fin["taux defaut"] = np.round((data_fin["defaut"]/data_fin["nombre"])*100,2)
    data_fin["minimum"] = list(df_1.iloc[:-1,0])
    data_fin["maximum"] = list(df_1.iloc[1:,0])

    print(plt.plot(data_fin["taux defaut"]))

    return data_fin

# Ce test permet de rendre compte du test du Khi deux 
def khi_deux(df_qual,target,alpha):
    
    liste_stat = []
    liste_pval = []
    for i in range(len(list(df_qual.columns))) : 
        table = pd.crosstab(df_qual[list(df_qual.columns)[i]],df_qual[target])
        stat, p, dof, expected = chi2_contingency(table)
        prob = 1 - alpha
        critical = chi2.ppf(prob, dof)
        liste_stat.append(stat)
        liste_pval.append(p)
        
    quali_1 = pd.DataFrame([liste_stat,liste_pval],index=["stat","p-value"],columns=df_qual.columns)
    quali_1 = quali_1.sort_values(by = "stat", axis=1,ascending = False)
    return quali_1

# Ces deux fonctions permettent de rendre compte de la stats de V de cramer par rapport à la target 

def cramers_v(x, y):
    """ Cette fonction permet à cramerV_join de fonctionner"""
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def cramerV_join(df_qual,target):
    """ * df_qual : la base de donées composée uniquement des variables qualitatives ;
        * target  : la variable cible """
    List_name = []
    Data=pd.DataFrame()

    # Création de la matrice du V de cramer
    
    for x in range (df_qual.shape[1]):
        for i in range (df_qual.shape[1]) :
            ListTest=[]
            ListTest.append(cramers_v(df_qual[list(df_qual.columns)[x]], df_qual[list(df_qual.columns)[i]]))
            List_name.append("{} et {}".format(list(df_qual.columns)[x],list(df_qual.columns)[i]))
            mySeries = pd.Series(ListTest,name=x)
            Data=pd.concat([Data,mySeries], axis=1)
    Data=Data.values
    
    data_cramer = pd.DataFrame(Data,index = ["CramerV"]) 
    data_cramer.columns = List_name
    #data_cramer = data_cramer.iloc[:, np.argsort(data_cramer.loc['CramerV'])] pour trier les sorties pas obliger
    
    print("importance de la dépendance par rapport à la variable du défaut, {} : ".format(target))
    
    return data_cramer

# Cette fonction permet de procéder aux étape de la régression logistique en step by step tout en assurant la parcimonie (BIC) et l'absence de corrélation jointe entre les variables explicatives suivant la stat du V de Cramer

def step_by_step_log(data,target):
    
    """ * data : C'est la base après tout le data processing réalisé ;
        * target : La variable cible"""
    
    df_khi_deux_tar = khi_deux(data,target,0.05)
    
    l = []
    l.append(target + " ~ ")
    bic_list = [100000000]
    col = []
    for i in range(1,len(df_khi_deux_tar.columns)) :
        l2 = []

        if i == 1 :
            for j in l :
                l2.append(j + df_khi_deux_tar.columns[i])
        else :
            for j in l :
                l2.append(j + " + " + df_khi_deux_tar.columns[i])

        col.append(df_khi_deux_tar.columns[i])
        c = cramerV_join(data[col],target)
        liste_corr = list(c.iloc[0,:])
        liste_corr_fin = list(filter(lambda x :(x < 0.7) or (x > 0.99),liste_corr))

        log = smf.logit(l2[0], data=data).fit()
        bic_list.append(log.bic)

        if (bic_list[-1] < bic_list[-2]) and (len(liste_corr_fin) == len(liste_corr)): # pb ici si la 2e cond n'est pas respecté il s'arrête à voir d'où ca vient
            l = l2
        else :
            col.pop(-1)

    col.append("loan_statu")
    bic_list = bic_list[1:]
    
    return l, col # list des variables sous format stats model et col sous format pandas


# Fonction de construction de la grille de score finale avec les valeurs associés à chaque modalité de nos variables dsicrètes.

def scorecard(data,var_keep,target,note_max = 1000):
    
    """ * data c'est la base utilisée dans la construction de la grille ;
        * var_keep c'est les colonnes de la base sans la target ;
        * target : la variable cible ;
        * Sur quelle note on souhaite évaluer nos données. """
  
    X = data[var_keep]
    y = data[target]
    
    list_col = list(X.columns)
    list_dum =[]

    X_dum = pd.get_dummies(data=X, columns=list_col)   

    classifier = LogisticRegression(random_state = 42)
    cv = StratifiedKFold(n_splits=4)

    liste_coef = []

    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X_dum.iloc[train,:], y.iloc[train,])
        coef = classifier.coef_
        liste_coef.append(coef)

    data_coef = pd.DataFrame(np.concatenate(liste_coef),columns=X_dum.columns)


    liste_mean = []
    for i in range(data_coef.shape[1]) :
        liste_mean.append(data_coef.iloc[:,i].mean())

    data_mean = pd.DataFrame(liste_mean).T
    data_mean.columns = data_coef.columns
    data_mean

    list_col_name = list(X.columns)

    list_group_var = []
    list_var_min = []
    list_var_sum_min = []
    list_var_max = []
    list_var_corr_fact = []

    for i in range(len(list_col_name)) :

        list_group_var.append([col for col in data_mean if col.startswith(list_col_name[i])])

        # j'ai la valeur absolue du minimum de chaque variable :

        value = list(data_mean[list_group_var[i]].iloc[0,:])

        list_var_min.append(max(value))

        # la liste avec les valeurs sommées du minimum de chaque variable en valeur absolue :

        list_var_sum_min.append(abs(data_mean[list_group_var[i]] - list_var_min[i]))

        # on prends le maximum de chaque variable afin de le sommer :

        list_var_max.append(max(list_var_sum_min[i].iloc[0,:]))

        # on créé alors le facteur de correction comme suit :

        correction_factor = note_max / sum(list_var_max)

    list_var_corr_fact = []

    for j in range(len(list_var_sum_min)):

        list_var_corr_fact.append(list_var_sum_min[j].iloc[0,:] * correction_factor)

    d = pd.DataFrame(pd.concat(list_var_corr_fact))
    d.columns = ["Note"]
    
    return d


# Fonction de construction du résumé obtenu dans la grille de scoer vaec les taux de défaut par modalité et les répartitions

def resume_score(data,var_keep,target):
    
    """ * data c'est la base utilisée dans la construction de la grille ;
        * var_keep c'est les colonnes de la base sans la target ;
        * target : la variable cible. """
    
    dt = scorecard(data,var_keep,target)

    l_value = [int(i) for i in list(map(lambda x : x[-1],list(dt.index)))]
    dt["Value"] = l_value

    list_var = [i[:-2] for i in list(dt.index)]
    list_var_fin = list(OrderedDict.fromkeys(list_var))

    def tx_def_tab(var):
        return pd.DataFrame(data[[var,target]].groupby(by=[var]).mean())

    l_finale = list(map(tx_def_tab,list_var_fin))

    flat_l = []
    flat_index = []
    for i in range(len(l_finale)):
        flat_l.append(list(l_finale[i][target]))
        flat_index.append(list(l_finale[i].index))

    flat_ls = []
    flat_indexs = []
    for i in flat_l:
        for j in i:
            flat_ls.append(j)
    for n in flat_index :
        for m in n :
            flat_indexs.append(m)

    dt["taux de défaut"] = flat_ls
    dt["correspondance"] = flat_indexs
    
    
    def repart_tab(var):
        return pd.DataFrame(df3[var].value_counts(normalize=True))

    l_fin_r = list(map(repart_tab,list_var_fin))

    flat_l_r = []
    flat_index_r = []
    for i in range(len(l_fin_r)):
        flat_l_r.append(list(l_fin_r[i][list_var_fin[i]].sort_index(ascending=True)))
        flat_index_r.append(list(l_fin_r[i].sort_index(ascending=True).index))

    flat_ls_r = []
    flat_indexs_r = []
    for i in flat_l_r:
        for j in i:
            flat_ls_r.append(j)
    for n in flat_index_r :
        for m in n :
            flat_indexs_r.append(m)

    dt["répartition"] = flat_ls_r
    dt["correspondance repartition"] = flat_indexs_r
    
    return dt
