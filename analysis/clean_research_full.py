"""Reproducible exploratory study. Clean input files are never modified."""
from pathlib import Path
import json,hashlib,sys,warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'outputs/fuel_cleaning_20260919'; OUT=ROOT/'outputs/clean_research_full_20260920'; T=OUT/'tables'; T.mkdir(exist_ok=True,parents=True)
NAMES={77:'Москва',50:'Московская область',78:'Санкт-Петербург',16:'Татарстан',66:'Свердловская область',54:'Новосибирская область',47:'Ленинградская область',52:'Нижегородская область',23:'Краснодарский край',63:'Самарская область',74:'Челябинская область',42:'Кемеровская область',72:'Тюменская область',2:'Башкортостан',24:'Красноярский край',55:'Омская область'}
DANGER=['Превышение скорости на 40-60 км/ч','Превышение скорости на 60-80 км/ч','Превышение скорости более чем на 80 км/ч','Проезд на красный сигнал светофора','Использование телефона за рулем','Движение по обочине','Выезд на полосу встречного движения или на трамвайные пути встречного направления','Не пропустил пешехода']
PERIODS={'pre':('2026-04-01','2026-05-25',54),'post':('2026-06-01','2026-09-01',92)}
def save(d,n): d.to_csv(T/(n+'.csv'),index=False);return d

def fit(formula,d,cluster='region',family=None):
 if family is None:
  m=smf.ols(formula,d,missing='raise').fit(cov_type='cluster',cov_kwds={'groups':d[cluster]},use_t=True)
 else:
  m=smf.glm(formula,d,family=family,offset=np.log(d.days/30),missing='raise').fit(cov_type='cluster',cov_kwds={'groups':d[cluster]},use_t=True)
 return m

def extract(m,label,term='post_new',unit='п.п.',scale=100,cluster='region'):
 ci=m.conf_int().loc[term]
 return dict(model=label,term=term,beta=m.params[term]*scale,se=m.bse[term]*scale,ci_low=ci.iloc[0]*scale,ci_high=ci.iloc[1]*scale,p_value=m.pvalues[term],n=int(m.nobs),unit=unit,cluster=cluster)

def wild_test(m,groups,term='post_new',B=4999):
 X=m.model.exog;y=m.model.endog;j=m.model.exog_names.index(term);n,k=X.shape
 X0=np.delete(X,j,axis=1); b0=np.insert(np.linalg.lstsq(X0,y,rcond=None)[0],j,0); e0=y-X@b0
 A=np.linalg.pinv(X.T@X);ug=np.unique(groups);G=len(ug)
 scores=np.array([X[groups==g].T@e0[groups==g] for g in ug]);xx=np.array([X[groups==g].T@X[groups==g] for g in ug]);rng=np.random.default_rng(260920);w=rng.choice([-1.,1.],(B,G))
 delta=w@scores@A; q=np.einsum('k,gkl->gl',A[j],xx); projected=w*(scores@A[j])[None,:]-delta@q.T
 se=np.sqrt((projected**2).sum(1)*(G/(G-1))*((n-1)/(n-k)))
 ts=delta[:,j]/se;actual=float(m.tvalues[term]);return (1+np.sum(np.abs(ts)>=abs(actual)))/(B+1)

def main():
 c=pd.read_csv(SRC/'clients_demographics_clean.csv',sep=';',decimal=',');fuel=pd.read_csv(SRC/'fuel_transaction_clean.csv',sep=';',decimal=',');fines=pd.read_csv(SRC/'fines_2026_clean.csv',sep=';')
 hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in SRC.glob('*clean.csv')}
 c['signup']=pd.to_datetime(c.subscription_creation_date);fuel['date']=pd.to_datetime(fuel.order_datetime);fines['date']=pd.to_datetime(fines.bill_offence_date)
 # Keep distinct client-vehicle pairs: one vehicle can be linked to several clients.
 v=c.sort_values('signup').drop_duplicates(['client_id','auto_document_id'])
 v['fines25']=v[['april_2025_fines','may_2025_fines','jun_2025_fines','jul_2025_fines','aug_2025_fines']].sum(axis=1)
 p=v.groupby('client_id').agg(region=('kladr_code','first'),gender=('gender','first'),age_type=('age_type_code','first'),auto_year=('auto_year','median'),price=('price','median'),cars=('auto_document_id','nunique'),fines25=('fines25','sum'))
 p['vehicle_age']=2026-p.auto_year;p['group']=np.select([p.vehicle_age.le(5),p.vehicle_age.ge(11)],['new','old'],default='middle');p['new']=(p.group=='new').astype(int)
 p['gender']=p.gender.fillna('unknown');p['age_type']=p.age_type.fillna('unknown');p['price_missing']=p.price.isna().astype(int);p['log_price']=np.log1p(p.price.fillna(p.price.median()));p['log_fines25']=np.log1p(p.fines25);p['region_name']=p.region.map(NAMES)
 p=p.reset_index(); keep=p[p.group.isin(['new','old'])].copy()
 fines['dangerous']=fines.offence_short_statement.isin(DANGER).astype(int);fines['broad']=((fines.dangerous==1)|fines.offence_short_statement.eq('Не пристегнут ремень безопасности')).astype(int)
 fines['speed40']=fines.offence_short_statement.isin(DANGER[:3]).astype(int);fines['large_amount']=(fines.total_fine_amount>=150000).astype(int)
 fines['period']=np.select([(fines.date>='2026-04-01')&(fines.date<'2026-05-25'),(fines.date>='2026-06-01')&(fines.date<'2026-09-01')],['pre','post'],default='excluded')
 yy=fines[fines.period!='excluded'].merge(keep,on='client_id',validate='many_to_one');yy['post']=(yy.period=='post').astype(int);yy['post_new']=yy.post*yy.new
 panel=[]
 for per,(start,end,days) in PERIODS.items():
  fx=fuel[(fuel.date>=start)&(fuel.date<end)].copy();fx['positive']=(fx.order_fuel_volume>0).astype(int);fx['positive_liters']=fx.order_fuel_volume.clip(lower=0);fx['spend']=fx.order_fuel_volume*fx.order_fuel_price_1liter
  fa=fx.groupby('client_id').agg(liters=('order_fuel_volume','sum'),positive_liters=('positive_liters','sum'),transactions=('positive','sum'),spend=('spend','sum'))
  ya=fines[fines.period==per].groupby('client_id').agg(all_fines=('bill_id','size'),dangerous_fines=('dangerous','sum'),broad_fines=('broad','sum'))
  x=p.set_index('client_id').join(fa).join(ya);cols=['liters','positive_liters','transactions','spend','all_fines','dangerous_fines','broad_fines'];x[cols]=x[cols].fillna(0)
  x['days']=days;x['period']=per;x['post']=int(per=='post');x['post_new']=x.post*x.new
  for col in cols:x[col+'_30']=x[col]*30/days
  x['dangerous_rate']=x.dangerous_fines_30*1000;x['all_rate']=x.all_fines_30*1000;x['active']=(x.transactions>0).astype(int)
  panel.append(x.reset_index())
 panel=pd.concat(panel,ignore_index=True);pk=panel[panel.group.isin(['new','old'])].copy()
 levels=pk.groupby(['group','period']).agg(clients=('client_id','size'),days=('days','first'),all_fines=('all_fines','sum'),dangerous_fines=('dangerous_fines','sum'),liters=('liters','sum'),transactions=('transactions','sum'),positive_liters=('positive_liters','sum'))
 levels['share']=100*levels.dangerous_fines/levels.all_fines
 for col in ['all_fines','dangerous_fines','liters','transactions','positive_liters']:levels[col+'_30']=levels[col]/levels.clients*30/levels.days
 save(levels.reset_index(),'prepost')
 profiles=p.groupby('group').agg(clients=('client_id','size'),median_age=('vehicle_age','median'),median_price=('price','median'),multi_car_share=('cars',lambda s:(s>1).mean()),female_share=('gender',lambda s:(s=='F').mean()),median_fines25=('fines25','median'));save(profiles.reset_index(),'profiles')
 miss=pd.DataFrame({'field':c.columns,'missing':c.isna().sum().values,'missing_pct':c.isna().mean().values*100});save(miss,'missingness')
 # Regional exposure from all clean clients, standardized AFTER averaging two z-scores.
 rg=panel.groupby(['region','region_name','period']).agg(clients=('client_id','size'),liters=('liters_30','mean'),transactions=('transactions_30','mean'),active_share=('active','mean')).reset_index()
 reg=rg.pivot(index=['region','region_name'],columns='period',values=['clients','liters','transactions','active_share']);reg.columns=['_'.join(t) for t in reg.columns];reg=reg.reset_index()
 for col in ['liters','transactions']:reg[col+'_change_pct']=100*(reg[col+'_post']/reg[col+'_pre']-1)
 z=lambda s:(s-s.mean())/s.std(ddof=0)
 reg['shortage']=z((z(-reg.liters_change_pct)+z(-reg.transactions_change_pct))/2)
 reg=save(reg,'region_exposure')
 yy=yy.merge(reg[['region','shortage']],on='region');pk=pk.merge(reg[['region','shortage']],on='region')
 controls='log_price + price_missing + log_fines25 + cars + C(gender) + C(age_type)'
 rhs='C(region)*post + C(region)*new + post_new + '+controls
 main=fit('dangerous ~ '+rhs,yy);rows=[extract(main,'Основная: регион × период + регион × группа')]
 rows.append(extract(fit('dangerous ~ '+rhs,yy,'client_id'),'Кластеризация по клиенту',cluster='client_id'))
 rows.append(extract(fit('dangerous ~ post*new',yy),'Без контролей','post:new'))
 rows.append(extract(fit('dangerous ~ C(region)*post + C(region)*new + post_new',yy),'Без демографических контролей'))
 rows.append(extract(fit('dangerous ~ '+rhs,yy[yy.cars==1]),'Только один автомобиль'))
 rows.append(extract(fit('dangerous ~ '+rhs,yy[yy.price_missing==0]),'Без пропущенной цены'))
 for col,label in [('broad','Опасные + непристёгнутый ремень'),('speed40','Только скорость 40+'),('large_amount','Штраф от 1 500 ₽')]: rows.append(extract(fit(col+' ~ '+rhs,yy),label))
 for cutoff in [3,7]:
  k=p[p.vehicle_age.le(cutoff)|p.vehicle_age.ge(11)].copy();k['new']=(k.vehicle_age<=cutoff).astype(int)
  a=fines[fines.period!='excluded'].merge(k,on='client_id');a['post']=(a.period=='post').astype(int);a['post_new']=a.post*a.new
  rows.append(extract(fit('dangerous ~ '+rhs,a),f'Новые ≤{cutoff} лет'))
 triple=fit('dangerous ~ '+rhs+' + post_new:shortage',yy);rows.append(extract(triple,'Региональная дозозависимость','post_new:shortage','п.п./SD'))
 save(pd.DataFrame(rows),'models');save(pd.DataFrame({'term':main.params.index,'beta':main.params.values,'se':main.bse.values,'p_value':main.pvalues.values,'ci_low':main.conf_int()[0].values,'ci_high':main.conf_int()[1].values}),'main_all_coefficients')
 print('MAIN',rows[0],flush=True)
 wild=wild_test(main,yy.region.to_numpy());print('WILD',wild,flush=True)
 # Individual fixed effects in two periods are first differences. Includes zero fine/purchase clients.
 w=pk.pivot(index='client_id',columns='period',values=['dangerous_rate','all_rate','liters_30','positive_liters_30','transactions_30','active'])
 w.columns=['_'.join(x) for x in w.columns];w=w.reset_index().merge(keep,on='client_id').merge(reg[['region','shortage']],on='region')
 mechanisms=[]
 for outcome,unit in [('dangerous_rate','штрафов / 1 000 клиентов / 30 дней'),('all_rate','штрафов / 1 000 клиентов / 30 дней'),('liters_30','л / клиент / 30 дней'),('positive_liters_30','л / клиент / 30 дней'),('transactions_30','заправок / клиент / 30 дней'),('active','доля клиентов')]:
  w['delta_'+outcome]=w[outcome+'_post']-w[outcome+'_pre'];f=fit('delta_'+outcome+' ~ new + C(region) + '+controls,w)
  mechanisms.append(extract(f,outcome,'new',unit,1))
  if outcome in ['liters_30','transactions_30']:
   f=fit('delta_'+outcome+' ~ new + new:shortage + C(region) + '+controls,w);mechanisms.append(extract(f,outcome+' × дефицит','new:shortage',unit+'/SD',1))
 save(pd.DataFrame(mechanisms),'mechanisms')
 poisson=fit('dangerous_fines ~ '+rhs,pk,family=sm.families.Poisson());pr=extract(poisson,'Poisson: опасные штрафы на клиента',unit='log(IRR)',scale=1);pr.update(irr=float(np.exp(pr['beta'])),irr_low=float(np.exp(pr['ci_low'])),irr_high=float(np.exp(pr['ci_high'])));save(pd.DataFrame([pr]),'poisson')
 # Region-specific regressions and raw counts. Client-clustered SE within each region.
 regional=[]
 for region,d in yy.groupby('region'):
  m=fit('dangerous ~ post*new + '+controls,d,'client_id');r=extract(m,NAMES[region],'post:new',cluster='client_id');r['region']=int(region)
  for group,new in [('new',1),('old',0)]:
   for per,post in [('pre',0),('post',1)]:
    q=d[(d.new==new)&(d.post==post)];r[f'{group}_{per}_n']=len(q);r[f'{group}_{per}_danger']=int(q.dangerous.sum());r[f'{group}_{per}_share']=100*q.dangerous.mean()
   r[group+'_change']=r[group+'_post_share']-r[group+'_pre_share']
  r['raw_did']=r['new_change']-r['old_change'];r['rare_events']=min(r[k] for k in ['new_pre_danger','new_post_danger','old_pre_danger','old_post_danger'])
  regional.append(r)
 rr=pd.DataFrame(regional);rr['p_holm']=multipletests(rr.p_value,method='holm')[1];rr=rr.merge(reg,on='region')
 rr['diagnosis']=np.where(rr.beta>0,'Обратный знак, индивидуальный ДИ проверяет точность',np.where(rr.ci_high>=0,'Отрицательный знак, ДИ включает ноль','Отрицательный эффект, индивидуальный ДИ ниже нуля'))
 save(rr,'regional_models')
 loo=[]
 for region in sorted(yy.region.unique()):loo.append(dict(excluded=NAMES[region],**extract(fit('dangerous ~ '+rhs,yy[yy.region!=region]),'Исключён регион')))
 save(pd.DataFrame(loo),'leave_one_region_out')
 # Monthly raw dynamics and event study with April baseline, excluding transition week.
 yy['month']=yy.date.dt.month
 evfit=fit('dangerous ~ C(region)*C(month) + C(region)*new + C(month):new + '+controls,yy)
 ev=[]
 for month in [5,6,7,8]:ev.append(dict(month=month,**extract(evfit,'Месячная динамика',f'C(month)[T.{month}]:new')))
 save(pd.DataFrame(ev),'event_study')
 pre=yy[yy.post==0].copy();pre['block']=pd.cut((pre.date-pd.Timestamp('2026-04-01')).dt.days,[-1,13,27,41,53],labels=[0,1,2,3]).astype(int)
 pref=fit('dangerous ~ C(region)*C(block) + C(region)*new + C(block):new + '+controls,pre)
 terms=[f'C(block)[T.{i}]:new' for i in [1,2,3]];R=np.zeros((3,len(pref.params)))
 for j,t in enumerate(terms):R[j,list(pref.params.index).index(t)]=1
 pre_p=float(pref.f_test(R).pvalue);save(pd.DataFrame([dict(block=i,**extract(pref,'Предтренд',term)) for i,term in enumerate(terms,1)]),'pretrends')
 monthly=[]
 for month,days in [(4,30),(5,24),(6,30),(7,31),(8,31)]:
  end='2026-05-25' if month==5 else str(pd.Timestamp(2026,month,1)+pd.offsets.MonthBegin(1))
  f=fuel[(fuel.date>=pd.Timestamp(2026,month,1))&(fuel.date<pd.Timestamp(end))].merge(keep,on='client_id');q=yy[yy.month==month]
  for group in ['new','old']:
   fq=f[f.group==group];yq=q[q.group==group];n=int((keep.group==group).sum());monthly.append(dict(month=month,group=group,days=days,clients=n,all_fines=len(yq),dangerous=int(yq.dangerous.sum()),share=100*yq.dangerous.mean(),dangerous_rate=1000*yq.dangerous.sum()/n*30/days,liters_30=fq.order_fuel_volume.sum()/n*30/days,transactions_30=(fq.order_fuel_volume>0).sum()/n*30/days,median_price=fq.loc[fq.order_fuel_volume>0,'order_fuel_price_1liter'].median()))
 save(pd.DataFrame(monthly),'monthly')
 # Full classification and group-period offence mix.
 classification=fines.groupby('offence_short_statement').agg(n=('bill_id','size'),dangerous=('dangerous','first'),broad=('broad','first'));save(classification.reset_index(),'classification')
 mix=yy.groupby(['group','period','offence_short_statement']).size().rename('n').reset_index();mix['share']=100*mix.n/mix.groupby(['group','period']).n.transform('sum');save(mix,'offence_mix')
 corr=w[['vehicle_age','price','cars','fines25','dangerous_rate_pre','dangerous_rate_post','liters_30_pre','liters_30_post','transactions_30_pre','transactions_30_post']].corr(method='spearman');save(corr.reset_index(),'correlations')
 save(p[['group','vehicle_age']].groupby(['group','vehicle_age']).size().rename('clients').reset_index(),'age_distribution')
 data_summary=dict(clean_rows={'clients':len(c),'fuel':len(fuel),'fines':len(fines)},clients=len(p),vehicles=v.auto_document_id.nunique(),pairs=len(v),duplicate_pairs=len(c)-len(v),shared_documents=int((v.groupby('auto_document_id').client_id.nunique()>1).sum()),groups=p.group.value_counts().to_dict(),fine_model_n=len(yy),fine_model_clients=yy.client_id.nunique(),panel_n=len(pk),regions=len(reg),wild_p=wild,wild_B=4999,pretrend_p=pre_p,main=rows[0],poisson=pr,hashes=hashes,unknown_prices=int(p.price_missing.sum()),regex_false_positives=int((fines.offence_short_statement.str.contains('пешеход',case=False)&~fines.offence_short_statement.isin(DANGER)).sum()),matrix_rank=int(np.linalg.matrix_rank(main.model.exog)),matrix_columns=main.model.exog.shape[1])
 (OUT/'summary.json').write_text(json.dumps(data_summary,ensure_ascii=False,indent=2));print(json.dumps(data_summary,ensure_ascii=False,indent=2),flush=True)
 # Internal data for reproducibility, kept local and excluded from presentation archives.
 w.to_pickle(ROOT/'tmp/clean_research_full/client_changes.pkl');p.to_pickle(ROOT/'tmp/clean_research_full/people.pkl');yy.to_pickle(ROOT/'tmp/clean_research_full/fines.pkl')
if __name__=='__main__':main()
