from clean_research_full import *
from scipy.special import expit
import patsy
p=pd.read_pickle(ROOT/'tmp/clean_research_full/people.pkl');yy=pd.read_pickle(ROOT/'tmp/clean_research_full/fines.pkl')
controls='log_price + price_missing + log_fines25 + cars + C(gender) + C(age_type)'
rhs='C(region)*post + C(region)*new + post_new + '+controls
m=smf.glm('dangerous ~ '+rhs,yy,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':yy.region})
# Standardize over distinct cohort clients, equal weight per client.
x=p[p.group.isin(['new','old'])].copy();eff=0;gradient=np.zeros(len(m.params))
for new,post,sign in [(1,1,1),(1,0,-1),(0,1,-1),(0,0,1)]:
 q=x.copy();q['new']=new;q['post']=post;q['post_new']=new*post
 X=np.asarray(patsy.build_design_matrices([m.model.data.design_info],q)[0]);pr=expit(X@m.params);eff+=sign*pr.mean();gradient+=sign*(pr*(1-pr))@X/len(X)
se=np.sqrt(gradient@m.cov_params()@gradient);crit=st.t.ppf(.975,15)
r=dict(model='Logit: стандартизированный DiD',term='marginal_DiD',beta=eff*100,se=se*100,ci_low=(eff-crit*se)*100,ci_high=(eff+crit*se)*100,p_value=2*st.t.sf(abs(eff/se),15),n=len(yy),unit='п.п.',cluster='region')
rows=[r]
# Same registration and offence region, based on actual names in the source.
name_alts={77:'Москва',50:'Московская область',78:'Санкт-Петербург',16:'Республика Татарстан',2:'Республика Башкортостан'}
def regmatch(r):
 key=name_alts.get(r.region,NAMES[r.region]);return key.lower().replace('республика ','') in str(r.region_name_x).lower().replace('республика ','')
# Merged fine frame has actual region_name_x and registered region_name_y.
print(yy.columns.tolist())
if 'region_name_x' in yy:
 match=yy.apply(regmatch,axis=1);d=yy[match];rows.append(extract(fit('dangerous ~ '+rhs,d),'Штраф в регионе регистрации'))
# Car-specific cohort on recorded offence vehicle, retains user-level controls.
c=pd.read_csv(SRC/'clients_demographics_clean.csv',sep=';',decimal=',').drop_duplicates(['client_id','auto_document_id'])
q=yy.drop(columns=['new','post_new']).merge(c[['client_id','auto_document_id','auto_year']].rename(columns={'auto_year':'specific_year'}),on=['client_id','auto_document_id'],how='inner',validate='many_to_one');q['specific_age']=2026-q.specific_year;q=q[(q.specific_age<=5)|(q.specific_age>=11)];q['new']=(q.specific_age<=5).astype(int);q['post_new']=q.post*q.new
rows.append(extract(fit('dangerous ~ '+rhs,q),'Возраст машины в конкретном штрафе'))
save(pd.DataFrame(rows),'extra_robustness')
print(pd.DataFrame(rows).to_string(index=False))
