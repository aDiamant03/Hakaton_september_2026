"""Regional robustness extension. Run with Python 3.9 and existing research_pydeps.
No source files or previous research outputs are modified.
"""
from pathlib import Path
import sys, json, hashlib, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Polygon, Patch

warnings.filterwarnings('ignore', category=FutureWarning)
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/regional_resilience_20260920'; T=OUT/'tables'; CHARTS=OUT/'charts'
SRC=ROOT/'outputs/fuel_cleaning_20260919'; PREV=ROOT/'outputs/clean_research_full_20260920'
sys.path.insert(0,str(ROOT/'analysis'))
from clean_research_full import DANGER, NAMES
from create_shortage_index_map import REGION_NAMES
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#344960','text.color':'#172d43','axes.edgecolor':'#b8c5d1','grid.color':'#dce4ec','svg.fonttype':'none'})
BLUE='#2878b7'; RED='#cb594b'; TEAL='#27877c'; GREY='#90a4b7'
manifest=[]
def save(d,n):d.to_csv(T/(n+'.csv'),index=False);return d
def z(x):return (x-x.mean())/x.std(ddof=0)
def chart(fig,name,title,note):
 fig.suptitle(title,x=.04,ha='left',fontsize=17,fontweight='bold',y=.99)
 fig.text(.04,.012,note,fontsize=8,color='#52677c',va='bottom')
 fig.savefig(CHARTS/(name+'.png'),dpi=165,bbox_inches='tight',facecolor='white')
 fig.savefig(CHARTS/(name+'.svg'),bbox_inches='tight',facecolor='white')
 plt.close(fig);manifest.append(dict(file=name,title=title,note=note))

def load():
 summary=json.loads((PREV/'summary.json').read_text())
 for name,h in summary['hashes'].items():assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==h
 c=pd.read_csv(SRC/'clients_demographics_clean.csv',sep=';',decimal=',');c['signup']=pd.to_datetime(c.subscription_creation_date)
 v=c.sort_values('signup').drop_duplicates(['client_id','auto_document_id'])
 v['fines25']=v[['april_2025_fines','may_2025_fines','jun_2025_fines','jul_2025_fines','aug_2025_fines']].sum(axis=1)
 p=v.groupby('client_id').agg(region=('kladr_code','first'),gender=('gender','first'),age_type=('age_type_code','first'),auto_year=('auto_year','median'),price=('price','median'),cars=('auto_document_id','nunique'),fines25=('fines25','sum'),signup=('signup','min')).reset_index()
 p['vehicle_age']=2026-p.auto_year;p['group']=np.select([p.vehicle_age.le(5),p.vehicle_age.ge(11)],['new','old'],default='middle');p['new']=(p.group=='new').astype(int)
 p['gender']=p.gender.fillna('unknown');p['age_type']=p.age_type.fillna('unknown');p['price_missing']=p.price.isna().astype(int)
 p['log_price']=np.log1p(p.price.fillna(p.price.median()));p['log_fines25']=np.log1p(p.fines25)
 fuel=pd.read_csv(SRC/'fuel_transaction_clean.csv',sep=';',decimal=',');fuel['date']=pd.to_datetime(fuel.order_datetime)
 fuel['positive_liters']=fuel.order_fuel_volume.clip(lower=0);fuel['transactions']=(fuel.order_fuel_volume>0).astype(int);fuel['spend']=fuel.positive_liters*fuel.order_fuel_price_1liter
 y=pd.read_csv(SRC/'fines_2026_clean.csv',sep=';');y['date']=pd.to_datetime(y.bill_offence_date);y['dangerous']=y.offence_short_statement.isin(DANGER).astype(int)
 assert len(p)==summary['clients']
 return p,fuel,y,summary

def panel(p,fuel,start='2026-04-01',pre_end='2026-05-25',post_start='2026-06-01',end='2026-09-01'):
 rows=[]
 for per,a,b in [('pre',start,pre_end),('post',post_start,end)]:
  f=fuel[fuel.date.between(a,b,inclusive='left')];days=(pd.Timestamp(b)-pd.Timestamp(a)).days
  g=f.groupby('client_id').agg(liters=('order_fuel_volume','sum'),positive_liters=('positive_liters','sum'),transactions=('transactions','sum'),spend=('spend','sum'))
  q=p.set_index('client_id').join(g);cols=['liters','positive_liters','transactions','spend'];q[cols]=q[cols].fillna(0)
  for col in cols:q[col+'_30']=q[col]*30/days
  q['period']=per;q['days']=days;rows.append(q.reset_index())
 return pd.concat(rows,ignore_index=True)

def region_metrics(pan):
 a=pan.groupby(['region','period']).agg(n=('client_id','size'),liters=('liters_30','mean'),positive_liters=('positive_liters_30','mean'),transactions=('transactions_30','mean'),spend=('spend','sum'),positive_total=('positive_liters','sum'))
 a['price']=a.spend/a.positive_total
 w=a.unstack();w.columns=['_'.join(x) for x in w.columns];w=w.reset_index()
 for col in ['liters','positive_liters','transactions','price']:w[col+'_change']=100*(w[col+'_post']/w[col+'_pre']-1)
 w['S']=z((z(-w.liters_change)+z(-w.transactions_change))/2)
 return w

def fit_fine(d,extra='',weights=None):
 rhs='C(region)*post + C(region)*new + post_new + log_price + price_missing + log_fines25 + cars + C(gender) + C(age_type)'+extra
 mod=smf.ols('dangerous ~ '+rhs,d) if weights is None else smf.wls('dangerous ~ '+rhs,d,weights=weights)
 return mod.fit(cov_type='cluster',cov_kwds={'groups':d.region},use_t=True)
def extract(m,label,term='post_new'):
 lo,hi=m.conf_int().loc[term]
 return dict(model=label,term=term,beta=100*m.params[term],ci_low=100*lo,ci_high=100*hi,p_value=m.pvalues[term],n=int(m.nobs))

def checked_wild(m,groups,term,B=4999):
 # Restricted wild cluster bootstrap, Rademacher weights, scaled full-rank design.
 X=m.model.exog.copy();y=m.model.endog;j=m.model.exog_names.index(term);n,k=X.shape
 scale=np.sqrt(np.mean(X**2,axis=0));X=X/scale
 assert np.linalg.matrix_rank(X)==k
 X0=np.delete(X,j,axis=1);b0=np.insert(np.linalg.lstsq(X0,y,rcond=None)[0],j,0);e0=y-X@b0
 pinv=np.linalg.pinv(X);A=pinv@pinv.T;ug=np.unique(groups);G=len(ug)
 scores=np.array([X[groups==g].T@e0[groups==g] for g in ug]);xx=np.array([X[groups==g].T@X[groups==g] for g in ug])
 rng=np.random.default_rng(260920);weights=rng.choice([-1.,1.],(B,G));delta=weights@scores@A
 q=np.einsum('k,gkl->gl',A[j],xx);projected=weights*(scores@A[j])[None,:]-delta@q.T
 se=np.sqrt((projected**2).sum(1)*(G/(G-1))*((n-1)/(n-k)));ts=delta[:,j]/se
 assert np.isfinite(ts).all() and (se>0).all(), 'Invalid bootstrap replications'
 return float((1+np.sum(np.abs(ts)>=abs(float(m.tvalues[term]))))/(B+1))

def calculate():
 p,fuel,y,summary=load();context=pd.read_csv(OUT/'sources/regional_context.csv')
 pan=panel(p,fuel);reg=region_metrics(pan).merge(context,on='region',validate='one_to_one')
 expected=pd.read_csv(PREV/'tables/region_exposure.csv').set_index('region')
 assert np.max(np.abs(reg.set_index('region').S-expected.shortage))<1e-10
 reg['population_m']=reg.population_2025/1e6;reg['grp_m']=reg.grp_pc_2024_rub/1e6;reg['sample_per100k']=reg.n_pre/reg.population_2025*1e5
 reg['logpop']=z(np.log(reg.population_2025));reg['loggrp']=z(np.log(reg.grp_pc_2024_rub));reg['refinery_z']=z(reg.refinery)
 reg['old_share']=reg.region.map(p.groupby('region').group.apply(lambda x:100*x.eq('old').mean()))
 reg['new_share']=reg.region.map(p.groupby('region').new.mean())*100
 reg['median_car_price']=reg.region.map(p.groupby('region').price.median())
 reg['drop']=-reg.liters_change
 rng=np.random.default_rng(260920)
 # Paired client bootstrap, zeros retained, uncertainty is conditional on service sample.
 w=pan.pivot(index='client_id',columns='period',values=['liters','positive_liters','transactions']);w.columns=['_'.join(t) for t in w.columns];w=w.join(p.set_index('client_id')[['region','group']])
 cis=[]
 for rid,d in w.groupby('region'):
  a=d[['liters_pre','liters_post']].to_numpy(); vals=[]
  for j in range(0,1000,100):
   ids=rng.integers(len(a),size=(100,len(a)));b=a[ids].sum(1);vals.extend(100*(b[:,1]/92/(b[:,0]/54)-1))
  cis.append(dict(region=rid,liters_ci_low=np.quantile(vals,.025),liters_ci_high=np.quantile(vals,.975)))
 reg=reg.merge(pd.DataFrame(cis),on='region')
 # Alternative definitions and population/cohort rules.
 variants={'База: все клиенты':reg.set_index('region').S,'Только литры':z(-reg.set_index('region').liters_change),'Только заправки':z(-reg.set_index('region').transactions_change),'Без возвратов':z((z(-reg.set_index('region').positive_liters_change)+z(-reg.set_index('region').transactions_change))/2)}
 keep_ids=w[(w.transactions_pre>0)&(w.transactions_post>0)].index
 alt_specs=[('Активны до и после',p[p.client_id.isin(keep_ids)],{}),('Подписка до 1 апреля',p[p.signup<'2026-04-01'],{}),('Апрель против июня',p,dict(pre_end='2026-05-01',end='2026-07-01')),('Апрель против августа',p,dict(pre_end='2026-05-01',post_start='2026-08-01'))]
 for name,pp,kw in alt_specs:variants[name]=region_metrics(panel(pp,fuel,**kw)).set_index('region').S
 vv=pd.DataFrame(variants);rank=vv.rank(ascending=False,method='min');save(rank.reset_index(),'rank_sensitivity')
 sensitivity=[]
 for name in vv:
  sensitivity.append(dict(definition=name,spearman=stats.spearmanr(vv.iloc[:,0],vv[name]).statistic,top4_overlap=len(set(vv.iloc[:,0].nlargest(4).index)&set(vv[name].nlargest(4).index))))
 save(pd.DataFrame(sensitivity),'index_sensitivity')
 # Monthly fuel per fixed client base, no spurious activity comparisons across unequal periods.
 monthly=[]
 ff=fuel.merge(p[['client_id','region']],on='client_id',validate='many_to_one')
 for month,days,end in [(4,30,'2026-05-01'),(5,24,'2026-05-25'),(6,30,'2026-07-01'),(7,31,'2026-08-01'),(8,31,'2026-09-01')]:
  a=ff[ff.date.between(f'2026-{month:02d}-01',end,inclusive='left')].groupby('region').agg(liters=('order_fuel_volume','sum'),tx=('transactions','sum'),price=('order_fuel_price_1liter','median'))
  a['n']=p.groupby('region').size();a['liters_30']=a.liters/a.n*30/days;a['tx_30']=a.tx/a.n*30/days;a['month']=month;monthly.append(a.reset_index())
 monthly=pd.concat(monthly);baseline=monthly[monthly.month==4].set_index('region').liters_30
 monthly['liters_index']=100*monthly.liters_30/monthly.region.map(baseline);save(monthly,'monthly_fuel')
 reg['aug_apr_index']=reg.region.map(monthly[monthly.month==8].set_index('region').liters_index)
 reg['june_apr_index']=reg.region.map(monthly[monthly.month==6].set_index('region').liters_index)
 # Demand decomposition: transactions/client/day times mean liters per transaction.
 reg['log_tx_change']=100*np.log(reg.transactions_post/reg.transactions_pre)
 reg['log_size_change']=100*np.log((reg.positive_liters_post/reg.transactions_post)/(reg.positive_liters_pre/reg.transactions_pre))
 assert np.max(abs(reg.log_tx_change+reg.log_size_change-100*np.log(reg.positive_liters_post/reg.positive_liters_pre)))<1e-9
 # External correlations: permutations across 16 regions and Holm for 5 hypotheses.
 corr=[]
 for col,label in [('population_m','Население'),('grp_m','ВРП на жителя'),('refinery','Наличие НПЗ'),('median_car_price','Цена автомобиля'),('old_share','Доля старых авто')]:
  x=stats.rankdata(reg[col]);a=stats.rankdata(reg['drop']);x=(x-x.mean())/np.linalg.norm(x-x.mean());a=(a-a.mean())/np.linalg.norm(a-a.mean());obs=float(x@a)
  perms=np.array([rng.permutation(a) for _ in range(19999)]);pv=(1+np.sum(abs(perms@x)>=abs(obs)-1e-12))/20000
  corr.append(dict(factor=label,column=col,rho=obs,p_permutation=pv))
 corr=pd.DataFrame(corr);corr['p_holm']=multipletests(corr.p_permutation,method='holm')[1];save(corr,'context_correlations')
 # Parsimonious regional regression, effective N=16, HC3.
 context_models=[]
 for label,form,d in [('Население + ВРП','drop ~ logpop + loggrp',reg),('Добавить НПЗ','drop ~ logpop + loggrp + refinery',reg),('Без двух столиц','drop ~ logpop + loggrp + refinery',reg[~reg.region.isin([77,78])]),('Без Башкортостана и Самары','drop ~ logpop + loggrp + refinery',reg[~reg.region.isin([2,63])]),('ВРП 2023 вместо 2024','drop ~ logpop + loggrp + refinery',reg.assign(loggrp=z(np.log(reg.grp_pc_2023_rub)))),('Без Тюмени и Кузбасса','drop ~ logpop + loggrp + refinery',reg[~reg.region.isin([72,42])])]:
  m=smf.ols(form,d).fit(cov_type='HC3',use_t=True)
  for term in m.params.index:
   lo,hi=m.conf_int().loc[term];context_models.append(dict(model=label,term=term,beta=m.params[term],ci_low=lo,ci_high=hi,p_value=m.pvalues[term],n=len(d),adj_r2=m.rsquared_adj))
 save(pd.DataFrame(context_models),'context_models')
 # Fine models recreated from clean rows, exact offence dictionary.
 y['period']=np.select([y.date.between('2026-04-01','2026-05-25',inclusive='left'),y.date.between('2026-06-01','2026-09-01',inclusive='left')],['pre','post'],default='excluded')
 yy=y[y.period!='excluded'].merge(p[p.group.isin(['new','old'])],on='client_id',validate='many_to_one')
 yy['post']=(yy.period=='post').astype(int);yy['post_new']=yy.post*yy.new
 yy=yy.merge(reg[['region','S','logpop','loggrp','refinery_z']],on='region',validate='many_to_one')
 main=fit_fine(yy); assert abs(100*main.params.post_new-summary['main']['beta'])<1e-7
 results=[extract(main,'Основная модель')]
 for label,dd in [('Без Москвы и Санкт-Петербурга',yy[~yy.region.isin([77,78])]),('Без Башкортостана и Самары',yy[~yy.region.isin([2,63])]),('Подписка до 1 апреля',yy[yy.signup<'2026-04-01'])]:
  results.append(extract(fit_fine(dd),label))
 for label,dd in [('Апрель против июня–августа',yy[(yy.post==1)|(yy.date<'2026-05-01')]),('Апрель против июня',yy[(yy.date<'2026-05-01')|((yy.date>='2026-06-01')&(yy.date<'2026-07-01'))])]:
  results.append(extract(fit_fine(dd),label))
 m=fit_fine(yy,weights=1/yy.groupby('region').region.transform('size'));results.append(extract(m,'Равный суммарный вес региона'))
 triple=fit_fine(yy,' + post_new:S');results.append(extract(triple,'Дозозависимость','post_new:S'))
 for label,extra in [('Доза + население',' + post_new:S + post_new:logpop'),('Доза + ВРП',' + post_new:S + post_new:loggrp'),('Доза + НПЗ',' + post_new:S + post_new:refinery_z'),('Доза + все 3 фактора',' + post_new:S + post_new:logpop + post_new:loggrp + post_new:refinery_z')]:
  results.append(extract(fit_fine(yy,extra),label,'post_new:S'))
 print('Fine models ready',flush=True)
 loo=[]
 for region in sorted(yy.region.unique()):
  m=fit_fine(yy[yy.region!=region],' + post_new:S');loo.append(dict(excluded=NAMES[region],**extract(m,'Исключение региона','post_new:S')))
 save(pd.DataFrame(loo),'dose_leave_one_region_out')
 # Cross-fit exposure: regional index built from middle-age cars, unused in fine contrast.
 independent=region_metrics(panel(p[p.group=='middle'],fuel)).set_index('region').S
 d=yy.copy();d['S']=d.region.map(independent);results.append(extract(fit_fine(d,' + post_new:S'),'Доза по авто 6–10 лет','post_new:S'))
 wild=checked_wild(triple,yy.region.to_numpy(),term='post_new:S',B=4999)
 main_wild=checked_wild(main,yy.region.to_numpy(),term='post_new',B=4999)
 results=pd.DataFrame(results);save(results,'fine_robustness')
 # Location mismatch, descriptive only: an offence location is not a fuel station location.
 same={77:'Москва',50:'Московская область',78:'Санкт-Петербург',16:'Республика Татарстан',66:'Свердловская область',54:'Новосибирская область',47:'Ленинградская область',52:'Нижегородская область',23:'Краснодарский край',63:'Самарская область',74:'Челябинская область',42:'Кемеровская область',72:'Тюменская область',2:'Республика Башкортостан',24:'Красноярский край',55:'Омская область'}
 fy=y[y.period!='excluded'].merge(p[['client_id','region']],on='client_id')
 fy['home']=fy.region.map(same);fy['same']=fy.region_name.eq(fy.home)
 save(fy.groupby(['home','region_name']).size().rename('fines').reset_index(),'fine_location_flows')
 reg['away_fine_share']=reg.region.map(fy.groupby('region')['same'].mean()).rsub(1)*100
 # Price robustness by matched client average unit price, not fuel-grade constant.
 cp=pan.pivot(index='client_id',columns='period',values=['spend','positive_liters']);cp.columns=['_'.join(t) for t in cp.columns]
 cp=cp[(cp.positive_liters_pre>0)&(cp.positive_liters_post>0)]
 cp['price_change']=100*((cp.spend_post/cp.positive_liters_post)/(cp.spend_pre/cp.positive_liters_pre)-1)
 cp=cp.join(p.set_index('client_id').region);reg['matched_price_change']=reg.region.map(cp.groupby('region').price_change.median())
 save(reg,'regional_results')
 quality=dict(clients=len(p),regions=len(reg),fine_rows=len(yy),input_hashes=summary['hashes'],reproduced_main_beta=100*main.params.post_new,main_wild_p=main_wild,dose_wild_p=wild,dose_wild_B=4999,bootstrap_B=1000,permutation_B=19999,refinery_regions=int(reg.refinery.sum()),no_refinery_regions=int((reg.refinery==0).sum()),same_region_fines_pct=100*fy.same.mean(),source_summary=summary)
 (OUT/'summary.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2))
 return reg,monthly,rank,results,corr,summary

def map_chart(reg,col,name,title,label,cmap='Blues',center=False):
 geo=json.loads((ROOT/'resources/geodata/candidates/russia_subjects_github.json').read_text());vals={REGION_NAMES[int(r.region)]:getattr(r,col) for r in reg.itertuples()}
 v=np.array(list(vals.values()));norm=TwoSlopeNorm(vmin=-max(abs(v)),vcenter=0,vmax=max(abs(v))) if center else Normalize(vmin=min(v),vmax=max(v));cm=plt.get_cmap(cmap)
 fig,ax=plt.subplots(figsize=(12.4,6.3));found=set()
 for f in geo['features']:
  k=f['properties'].get('NL_NAME_1');value=vals.get(k);color=cm(norm(value)) if value is not None else '#e7edf2'
  if value is not None:found.add(k)
  g=f['geometry'];polys=[g['coordinates']] if g['type']=='Polygon' else g['coordinates']
  for poly in polys:
   a=np.array(poly[0]);ax.add_patch(Polygon(a,closed=True,facecolor=color,edgecolor='white',lw=.45))
 for rid,label_,xy in [(77,'77',(37.62,55.75)),(78,'78',(30.32,59.94))]:
  value=float(reg.set_index('region').loc[rid,col]);ax.scatter(*xy,s=70,color=cm(norm(value)),edgecolor='#24394d',zorder=6);ax.annotate(label_,xy,xytext=(4,5),textcoords='offset points',fontsize=8,zorder=7)
 for r in reg.itertuples():
  if r.region not in [77,78]:
   xy={50:(40.3,55.15),47:(33.5,60.25)}.get(r.region,(r.lon,r.lat))
   ax.annotate(str(r.region),xy,xytext=(3,3),textcoords='offset points',fontsize=8,zorder=5,bbox=dict(fc='white',alpha=.7,ec='none',pad=.5))
 ax.set_xlim(19,116);ax.set_ylim(43,73);ax.set_aspect(1.65);ax.axis('off');assert found==set(vals)
 cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),ax=ax,shrink=.65,pad=.01);cb.set_label(label)
 if col=='refinery':cb.set_ticks([0,1]);cb.set_ticklabels(['Нет в перечне','Есть НПЗ'])
 ax.legend(handles=[Patch(facecolor='#e7edf2',label='Вне выборки')],loc='lower left',frameon=False)
 fig.subplots_adjust(top=.88,bottom=.09,left=.02,right=.94)
 chart(fig,name,title,'16 регионов. Числа — коды КЛАДР. Москва и Санкт-Петербург выделены точками. Фрагмент карты, не вся Россия.')

def plots(reg,monthly,rank,results,corr,summary):
 r=reg.sort_values('drop',ascending=False);labels=r.region_name.tolist();idx=np.arange(len(r))
 fig,ax=plt.subplots(figsize=(12,7));ax.barh(idx,r.liters_change,color=[RED if x>30 else BLUE for x in r['drop']],alpha=.85)
 ax.errorbar(r.liters_change,idx,xerr=[r.liters_change-r.liters_ci_low,r.liters_ci_high-r.liters_change],fmt='none',ecolor='#253b50',capsize=3)
 ax.set_yticks(idx,labels);ax.invert_yaxis();ax.axvline(0,color=GREY,lw=1);ax.set_xlabel('Изменение литров на клиента и 30 дней, %')
 for i,v in enumerate(r.liters_change):ax.text(v-1 if v<0 else v+1,i,f'{v:+.1f}%',ha='right' if v<0 else 'left',va='center',fontsize=9)
 ax.set_xlim(-85,20);fig.subplots_adjust(left=.24,top=.9,bottom=.12)
 chart(fig,'01_fuel_decline','Падение покупок различается в разы','До: 01.04–24.05; после: 01.06–31.08. 95% ДИ: 1 000 повторных выборок клиентов внутри региона.')
 map_chart(reg,'drop','02_map_decline','География снижения покупок','Снижение литров, %','OrRd',False)
 map_chart(reg,'population_m','03_map_population','Масштаб регионального спроса','Население 01.01.2025, млн')
 map_chart(reg,'grp_m','04_map_economy','Экономика до кризиса','ВРП на жителя 2024, млн ₽')
 map_chart(reg,'refinery','05_map_refineries','Наличие крупного НПЗ в регионе','0 = вне перечня; 1 = есть','YlGnBu')
 fig,ax=plt.subplots(figsize=(12,7));heat=monthly.pivot(index='region',columns='month',values='liters_index').loc[r.region];im=ax.imshow(heat,vmin=25,vmax=115,cmap='RdYlBu')
 ax.set_yticks(idx,labels);ax.set_xticks(range(5),['Апрель','Май 1–24','Июнь','Июль','Август'])
 for i in range(16):
  for j in range(5):ax.text(j,i,f'{heat.iloc[i,j]:.0f}',ha='center',va='center',fontsize=9,color='#13283a')
 fig.colorbar(im,ax=ax,label='Апрель = 100');fig.subplots_adjust(left=.25,top=.9,bottom=.1)
 chart(fig,'06_monthly_heatmap','Когда возник спад и есть ли восстановление','Фиксированный состав клиентов; поправка на число дней. Май ограничен 24-м числом. Индексы не очищены от сезонности.')
 fig,axes=plt.subplots(4,4,figsize=(13,10),sharex=True,sharey=True)
 for ax,row in zip(axes.flat,r.itertuples()):
  d=monthly[monthly.region==row.region];ax.plot(d.month,d.liters_index,'o-',color=BLUE);ax.axhline(100,color=GREY,lw=.7);ax.axvline(5.5,color=RED,lw=.8,ls='--');ax.set_title(row.region_name.replace(' область',' обл.'),fontsize=9);ax.set_xticks([4,5,6,7,8],['А','М','И','И','А']);ax.set_ylim(10,125);ax.grid(axis='y',alpha=.5)
 fig.subplots_adjust(top=.91,bottom=.08,hspace=.48,wspace=.17)
 chart(fig,'07_regional_paths','16 отдельных траекторий покупок','Литры на клиента и 30 дней. Апрель = 100. Вертикальная линия: начало июня; отсутствие восстановления ≠ остановка НПЗ.')
 fig,ax=plt.subplots(figsize=(12,7));ax.barh(idx-.17,r.log_tx_change,height=.32,color=BLUE,label='Частота заправок');ax.barh(idx+.17,r.log_size_change,height=.32,color='#65b3ca',label='Литры на заправку');ax.axvline(0,color=GREY,lw=.7)
 ax.set_yticks(idx,labels);ax.invert_yaxis();ax.set_xlabel('Вклад в изменение: 100 × логарифм отношения');ax.legend(loc='lower left');fig.subplots_adjust(left=.24,top=.9,bottom=.12)
 chart(fig,'08_decomposition','Снижение частоты или размера заправки','Точное разложение положительных литров: ln(Lпосле/Lдо) = ln(Tпосле/Tдо) + ln(размерпосле/размердо). Это лог-пункты, не обычные %.')
 fig,axes=plt.subplots(1,2,figsize=(12.5,6))
 for ax,col,xlab in zip(axes,['population_m','grp_m'],['Население, млн (2025)','ВРП на жителя, млн ₽ (2024)']):
  for flag,mark,color in [(0,'o',BLUE),(1,'s',RED)]:
   d=reg[reg.refinery==flag];ax.scatter(d[col],d['drop'],s=75,marker=mark,c=color,label='Есть НПЗ' if flag else 'Нет в перечне')
  for row in reg.itertuples():ax.annotate(str(row.region),(getattr(row,col),row.drop),xytext=(3,4),textcoords='offset points',fontsize=8)
  ax.set_xlabel(xlab);ax.set_ylabel('Снижение покупок, %');ax.grid(alpha=.35);ax.legend(fontsize=8)
 fig.subplots_adjust(top=.86,bottom=.15,wspace=.26)
 chart(fig,'09_context_scatter','Население, экономика и присутствие НПЗ','Каждая точка — регион, а не клиент. Коды КЛАДР расшифрованы в региональных карточках. Корреляция не доказывает механизм.')
 fig,ax=plt.subplots(figsize=(12,7));ax.barh(idx,r.sample_per100k,color=BLUE);ax.set_yticks(idx,labels);ax.invert_yaxis();ax.set_xlabel('Клиентов в выборке на 100 000 жителей')
 for i,row in enumerate(r.itertuples()):ax.text(row.sample_per100k+1,i,f'n={int(row.n_pre):,}'.replace(',',' '),va='center',fontsize=9)
 fig.subplots_adjust(left=.24,top=.9,bottom=.12);chart(fig,'10_sample_coverage','Где выборка представляет население слабее','Население не заменяет число наблюдаемых клиентов. Эти значения не являются оценкой доли рынка банка.')
 fig,ax=plt.subplots(figsize=(10,5));ax.barh(corr.factor,corr.rho,color=[RED if x>0 else BLUE for x in corr.rho]);ax.axvline(0,color=GREY);ax.set_xlim(-1,1);ax.set_xlabel('Ранговая корреляция со снижением покупок')
 for i,row in enumerate(corr.itertuples()):ax.text(.98,i,f'ρ={row.rho:+.2f}; p Holm={row.p_holm:.3f}',ha='right',va='center',fontsize=9)
 fig.subplots_adjust(left=.23,top=.86,bottom=.16);chart(fig,'11_correlations','Какие связи видны на 16 регионах','19 999 перестановок регионов; поправка Holm на 5 сравнений. Это исследовательские, не причинные проверки.')
 fig,ax=plt.subplots(figsize=(10,5));a=reg[reg.refinery==1];b=reg[reg.refinery==0]
 for i,d in enumerate([b,a]):
  jitter=np.linspace(-.16,.16,len(d));ax.scatter(i+jitter,d['drop'],s=80,c=BLUE if i==0 else RED)
  for xx,row in zip(i+jitter,d.itertuples()):ax.annotate(str(row.region),(xx,row.drop),xytext=(3,3),textcoords='offset points',fontsize=8)
  ax.plot([i-.22,i+.22],[d['drop'].median()]*2,color='#182c41',lw=3)
 ax.set_xticks([0,1],[f'Нет в перечне (n={len(b)})',f'Есть крупный НПЗ (n={len(a)})']);ax.set_ylabel('Снижение покупок, %');ax.set_xlim(-.5,1.5);fig.subplots_adjust(top=.84,bottom=.16)
 chart(fig,'12_refinery_comparison','Собственный НПЗ не гарантирует стабильные покупки','Чёрная черта — медиана. Сравнение не учитывает загрузку, ассортимент, контракты и межрегиональные поставки.')
 fig,ax=plt.subplots(figsize=(12,7));ax.scatter(r.price_change,r.liters_change,s=70,c=r.refinery,cmap='coolwarm');ax.axhline(0,color=GREY);ax.axvline(0,color=GREY)
 for row in r.itertuples():ax.annotate(str(row.region),(row.price_change,row.liters_change),xytext=(4,3),textcoords='offset points')
 ax.set_xlabel('Изменение цены: средняя с весом по литрам, %');ax.set_ylabel('Изменение покупок, %');ax.grid(alpha=.3);fig.subplots_adjust(top=.88,bottom=.14)
 chart(fig,'13_price_volume','Цена и объём: две разные стороны изменения','В датасете нет марки топлива и АЗС. Изменение средней цены включает изменение состава покупок.')
 fig,ax=plt.subplots(figsize=(12.5,7));a=rank.loc[r.region];im=ax.imshow(a,cmap='Blues_r',vmin=1,vmax=16,aspect='auto');ax.set_yticks(idx,labels);ax.set_xticks(range(len(a.columns)),a.columns,rotation=28,ha='right',fontsize=8)
 for i in range(16):
  for j in range(len(a.columns)):ax.text(j,i,str(int(a.iloc[i,j])),ha='center',va='center',color='white' if a.iloc[i,j]<7 else '#18324b',fontsize=9)
 fig.subplots_adjust(left=.24,top=.9,bottom=.21);chart(fig,'14_rank_robustness','Меняется ли порядок регионов при смене расчёта','1 — самый сильный спад. Активные до и после — отобранная по поведению группа, поэтому это только чувствительность.')
 rr=pd.read_csv(PREV/'tables/regional_models.csv').sort_values('beta');fig,ax=plt.subplots(figsize=(12,7));ax.errorbar(rr.beta,np.arange(16),xerr=[rr.beta-rr.ci_low,rr.ci_high-rr.beta],fmt='o',color=BLUE,capsize=3);ax.axvline(0,color=RED,ls='--');ax.set_yticks(range(16),rr.region_name);ax.invert_yaxis();ax.set_xlabel('Изменение доли опасных штрафов: новые минус старые, п.п.');fig.subplots_adjust(left=.24,top=.9,bottom=.12)
 chart(fig,'15_regional_effects','Отдельные региональные эффекты оценены неточно','95% ДИ с кластеризацией по клиенту; исход — доля среди выписанных штрафов. Для выводов по регионам проверяем Holm.')
 loo=pd.read_csv(PREV/'tables/leave_one_region_out.csv');fig,ax=plt.subplots(figsize=(12,7));ax.errorbar(loo.beta,idx,xerr=[loo.beta-loo.ci_low,loo.ci_high-loo.beta],fmt='o',color=BLUE,capsize=3);ax.axvline(0,color=RED,ls='--');ax.set_yticks(idx,loo.excluded);ax.invert_yaxis();ax.set_xlabel('Основной эффект после исключения региона, п.п.');fig.subplots_adjust(left=.24,top=.9,bottom=.12)
 chart(fig,'16_leave_one_out','Основной результат: исключаем регионы по одному','Каждая строка — новая модель без указанного региона. Регион × период и регион × группа; ошибки по регионам.')
 a=results[results.term=='post_new'];fig,ax=plt.subplots(figsize=(12,6));ii=np.arange(len(a));ax.errorbar(a.beta,ii,xerr=[a.beta-a.ci_low,a.ci_high-a.beta],fmt='o',color=BLUE,capsize=4);ax.axvline(0,color=RED,ls='--');ax.set_yticks(ii,a.model);ax.invert_yaxis();ax.set_xlabel('Эффект, п.п.');fig.subplots_adjust(left=.34,top=.86,bottom=.16)
 chart(fig,'17_fine_robustness','Основной эффект при смене географии и весов','Основная модель воспроизведена по исходным clean-строкам. Равный вес региона меняет целевую совокупность.')
 a=results[results.term=='post_new:S'];fig,ax=plt.subplots(figsize=(12,5.5));ii=np.arange(len(a));ax.errorbar(a.beta,ii,xerr=[a.beta-a.ci_low,a.ci_high-a.beta],fmt='o',color=TEAL,capsize=4);ax.axvline(0,color=RED,ls='--');ax.set_yticks(ii,a.model);ax.invert_yaxis();ax.set_xlabel('Усиление эффекта на 1 SD индекса спада, п.п.');fig.subplots_adjust(left=.3,top=.88,bottom=.15)
 chart(fig,'18_dose_controls','Дозозависимость: выдерживает ли внешние факторы','Добавлены взаимодействия Post × New с населением, ВРП и НПЗ. Постоянные уровни уже поглощены региональными эффектами.')
 a=pd.read_csv(T/'dose_leave_one_region_out.csv');fig,ax=plt.subplots(figsize=(12,7));ax.errorbar(a.beta,idx,xerr=[a.beta-a.ci_low,a.ci_high-a.beta],fmt='o',color=TEAL,capsize=3);ax.axvline(0,color=RED,ls='--');ax.set_yticks(idx,a.excluded);ax.invert_yaxis();ax.set_xlabel('Дозозависимость после исключения региона, п.п. / SD');fig.subplots_adjust(left=.24,top=.9,bottom=.12)
 chart(fig,'19_dose_leave_one_out','Какие регионы определяют дозозависимость','Индекс S зафиксирован по полной выборке для сопоставимости коэффициентов. 95% ДИ, кластеризация по регионам.')
 fig,ax=plt.subplots(figsize=(12,5));ev=pd.read_csv(PREV/'tables/event_study.csv');ax.errorbar(ev.month,ev.beta,yerr=[ev.beta-ev.ci_low,ev.ci_high-ev.beta],fmt='o-',capsize=5,color=BLUE);ax.scatter(4,0,color=GREY);ax.axhline(0,color=RED,ls='--');ax.axvline(5.5,color=GREY);ax.set_xticks([4,5,6,7,8],['Апрель: база','Май 1–24','Июнь','Июль','Август']);ax.set_ylabel('Разница новые–старые относительно апреля, п.п.');fig.subplots_adjust(top=.87,bottom=.15)
 chart(fig,'20_event_study','Изменение состава штрафов по месяцам','Май — короткая проверка до периода после. Отсутствие значимого предтренда не доказывает параллельные тренды.')
 fig,ax=plt.subplots(figsize=(12,7));ax.barh(idx,r.away_fine_share,color=GREY);ax.set_yticks(idx,labels);ax.invert_yaxis();ax.set_xlabel('Доля штрафов вне региона регистрации, %');fig.subplots_adjust(left=.24,top=.9,bottom=.12)
 chart(fig,'21_location_mismatch','Регистрация клиента и место нарушения','Описательная проверка мобильности. Место нарушения не устанавливает место заправки и не восстанавливает логистику топлива.')
 # Generic normalized supply stress, intentionally not calibrated to unobserved actual supply shares.
 fig,axes=plt.subplots(1,2,figsize=(12,5.5));loss=np.linspace(0,.6,61);buffer=np.linspace(0,.8,41);h=(1-loss[None,:]*(1-buffer[:,None]))*100
 im=axes[0].imshow(h,origin='lower',extent=[0,60,0,80],aspect='auto',cmap='RdYlGn',vmin=40,vmax=100);axes[0].set_xlabel('Потеря исходных поставок, %');axes[0].set_ylabel('Доля потери, замещённая резервом, %');fig.colorbar(im,ax=axes[0],label='Сохранившееся предложение, %')
 for stock in [3,7,14]:
  delay=np.arange(0,22);availability=100*(1-.3*np.maximum(delay-stock,0)/30);axes[1].plot(delay,availability,label=f'Запас {stock} дней')
 axes[1].set_xlabel('Задержка восстановления / довоза, дней');axes[1].set_ylabel('Обеспеченность за 30 дней, %');axes[1].legend();axes[1].grid(alpha=.3);fig.subplots_adjust(top=.83,bottom=.17,wspace=.34)
 chart(fig,'22_supply_stress','Стресс-тест снабжения: роль замещения и запасов','СЦЕНАРИЙ, не оценка регионов. Слева: 1−q(1−b). Справа: q=30%, потери начинаются после исчерпания заданного запаса.')
 fig,ax=plt.subplots(figsize=(12,6));r2=reg.sort_values('drop');ax.scatter(r2.june_apr_index,r2.aug_apr_index,s=80,c=r2.refinery,cmap='coolwarm');ax.plot([20,120],[20,120],ls='--',color=GREY)
 for row in r2.itertuples():ax.annotate(str(row.region),(row.june_apr_index,row.aug_apr_index),xytext=(4,3),textcoords='offset points')
 ax.set_xlabel('Июнь / апрель × 100');ax.set_ylabel('Август / апрель × 100');ax.grid(alpha=.3);fig.subplots_adjust(top=.88,bottom=.14)
 chart(fig,'23_recovery','К августу: восстановление или углубление спада','Выше диагонали — покупки в августе выше июня. Это устойчивость наблюдаемых покупок, не доказанное восстановление мощности НПЗ.')
 a=pd.read_csv(T/'context_models.csv');a=a[a.term=='refinery'];fig,ax=plt.subplots(figsize=(12,5));ii=np.arange(len(a));ax.errorbar(a.beta,ii,xerr=[a.beta-a.ci_low,a.ci_high-a.beta],fmt='o',color=RED,capsize=4);ax.axvline(0,color=GREY,ls='--');ax.set_yticks(ii,a.model);ax.invert_yaxis();ax.set_xlabel('Разница спада при наличии НПЗ, п.п.');fig.subplots_adjust(left=.34,top=.86,bottom=.16)
 chart(fig,'24_refinery_adjusted','НПЗ после учёта населения и экономики','OLS по регионам, ошибки HC3. N=12–16. Интервалы показывают неопределённость; коэффициент не является причинным эффектом НПЗ.')
 fig,ax=plt.subplots(figsize=(12,5.5))
 events=[('Кириши: сообщение об остановке','2026-05-05',BLUE),('Москва: сообщение об остановке','2026-05-17',BLUE),('Сызрань: сообщение об остановке','2026-05-21',BLUE),('Куйбышевский: поисковый индекс','2026-06-10',GREY),('Уфа: сообщения сторон расходятся','2026-06-25',RED)]
 for i,(label,day,color) in enumerate(events):
  dt=pd.Timestamp(day);ax.scatter(dt,i,s=100,color=color);ax.annotate(dt.strftime('%d.%m'),(dt,i),xytext=(8,5),textcoords='offset points',fontsize=10)
 ax.set_yticks(range(5),[x[0] for x in events]);ax.invert_yaxis();ax.axvspan(pd.Timestamp('2026-04-01'),pd.Timestamp('2026-05-25'),color=BLUE,alpha=.07);ax.axvspan(pd.Timestamp('2026-06-01'),pd.Timestamp('2026-09-01'),color=RED,alpha=.07)
 ax.axvline(pd.Timestamp('2026-06-01'),color=RED,ls='--');ax.set_xlim(pd.Timestamp('2026-04-01'),pd.Timestamp('2026-08-31'))
 ax.set_xticks(pd.to_datetime(['2026-04-01','2026-05-01','2026-06-01','2026-07-01','2026-08-01']),['01.04','01.05','01.06','01.07','01.08']);fig.subplots_adjust(left=.4,top=.86,bottom=.16)
 chart(fig,'25_event_timing','Остановки не совпадают с одной общей датой','Выборочная хронология СМИ, не полный реестр. Даты восстановления неизвестны; сообщение об атаке не равно подтверждённому простою.')
 (OUT/'chart_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':
 if '--plots-only' in sys.argv:
  args=(pd.read_csv(T/'regional_results.csv'),pd.read_csv(T/'monthly_fuel.csv'),pd.read_csv(T/'rank_sensitivity.csv').set_index('region'),pd.read_csv(T/'fine_robustness.csv'),pd.read_csv(T/'context_correlations.csv'),json.loads((PREV/'summary.json').read_text()))
 else:args=calculate()
 plots(*args);print('DONE: 25 figures',flush=True)
