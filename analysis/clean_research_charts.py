from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/clean_research_full_20260920';D=OUT/'tables';C=OUT/'charts';C.mkdir(exist_ok=True)
BLUE='#1769C2';SKY='#74BBEC';NAVY='#123F78';PALE='#E8F4FE';INK='#133453';MUTED='#526F8B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':19,'axes.titleweight':'bold','axes.titlecolor':INK,'axes.labelcolor':INK,'text.color':INK,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#C3D6E6','grid.color':'#DCEAF5','svg.fonttype':'none','savefig.facecolor':'white'})
cmap=LinearSegmentedColormap.from_list('study_blues',['#D9EFFD','#74BBEC','#1769C2','#10396E'])
manifest=[];native=[]
def read(n):return pd.read_csv(D/(n+'.csv'))
def finish(fig,ax,name,title,x='',y='',note='',grid=True):
 ax.set_title(title,loc='left',pad=22);ax.set_xlabel(x);ax.set_ylabel(y)
 if grid:ax.grid(axis='y',alpha=.65);ax.set_axisbelow(True)
 if note:fig.text(.02,.012,note,fontsize=10,color=MUTED)
 fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(C/(name+'.png'),dpi=180);fig.savefig(C/(name+'.svg'));plt.close(fig);manifest.append({'id':name,'title':title,'note':note})
def chartdata(name,typ,cats,series,unit):native.append(dict(id=name,type=typ,categories=list(cats),series=series,unit=unit))
def forest(d,name,title,label='model',xlabel='Разница изменений, п.п.',scale=1):
 fig,ax=plt.subplots(figsize=(12,max(4.8,len(d)*.42+1.5)));y=np.arange(len(d));ax.errorbar(d.beta*scale,y,xerr=np.array([d.beta-d.ci_low,d.ci_high-d.beta])*scale,fmt='o',color=BLUE,ecolor=SKY,lw=2.4,capsize=3);ax.axvline(0,color=MUTED,ls='--');ax.set_yticks(y,d[label]);ax.invert_yaxis();finish(fig,ax,name,title,x=xlabel,note='Точки — оценки. Отрезки — 95% доверительные интервалы.',grid=False)
lev=read('prepost');monthly=read('monthly');reg=read('region_exposure');rr=read('regional_models');prof=read('profiles');s=json.load(open(OUT/'summary.json'))
prof['label']=prof.group.map({'new':'Новые 0–5 лет','middle':'Средние 6–10 лет','old':'Старые 11+ лет'});prof=prof.set_index('group').loc[['new','middle','old']].reset_index()
fig,ax=plt.subplots(figsize=(10,5.5));bars=ax.bar(prof.label,prof.clients,color=[BLUE,SKY,NAVY]);ax.bar_label(bars,fmt='%.0f',padding=4);finish(fig,ax,'01_groups','14 437 клиентов в очищенной выборке',y='Клиенты',note='Основное сравнение: 3 240 новых и 7 097 старых автомобилей.')
chartdata('01_groups','bar',prof.label,[dict(name='Клиенты',values=prof.clients.tolist(),fill=BLUE)],'Клиенты')
age=read('age_distribution').groupby('vehicle_age').clients.sum();fig,ax=plt.subplots(figsize=(11,5.3));ax.bar(age.index,age,color=BLUE);ax.axvline(5.5,color=SKY,ls='--');ax.axvline(10.5,color=NAVY,ls='--');finish(fig,ax,'02_age','Возраст автомобиля задаёт контрастные группы','Медианный возраст автомобилей клиента, лет','Клиенты')
months=['Апрель','Май 1–24','Июнь','Июль','Август']
for col,name,title,unit in [('liters_30','03_liters','Покупки топлива снижаются в обеих группах','л / клиент / 30 дней'),('transactions_30','04_transactions','Частота положительных заправок','Заправки / клиент / 30 дней'),('share','05_monthly_share','Доля опасных штрафов по месяцам','% от всех штрафов'),('dangerous_rate','06_monthly_rate','Частота опасных штрафов по месяцам','На 1 000 клиентов / 30 дней'),('median_price','07_fuel_price','Медианная цена положительной заправки','₽ / литр')]:
 fig,ax=plt.subplots(figsize=(11,5.5));series=[]
 for group,color,label in [('new',BLUE,'Новые'),('old',SKY,'Старые')]:
  a=monthly[monthly.group==group].sort_values('month');ax.plot(range(5),a[col],color=color,lw=3,marker='o',ms=7,label=label);series.append(dict(name=label,values=a[col].tolist(),fill=color))
 ax.set_xticks(range(5),months);ax.axvline(1.5,color=MUTED,ls='--');ax.legend(frameon=False);finish(fig,ax,name,title,y=unit,note='25–31 мая исключены. Показатели частоты приведены к 30 дням.');chartdata(name,'line',months,series,unit)
for col,name,title,unit,mult in [('share','08_share_prepost','Опасная доля: новые 7,47% → 5,02%','% от всех штрафов',1),('dangerous_fines_30','09_rate_prepost','Опасные штрафы на клиента','На 1 000 клиентов / 30 дней',1000),('all_fines_30','10_all_fines','Общая частота штрафов не повторяет опасную долю','На 1 000 клиентов / 30 дней',1000)]:
 fig,ax=plt.subplots(figsize=(10,5.7));x=np.arange(2);series=[]
 for i,(per,color,label) in enumerate([('pre',SKY,'До'),('post',BLUE,'После')]):
  a=lev[lev.period==per].set_index('group').loc[['new','old']][col].values*mult;b=ax.bar(x+(i-.5)*.3,a,width=.3,color=color,label=label);ax.bar_label(b,fmt='%.2f',padding=4);series.append(dict(name=label,values=a.tolist(),fill=color))
 ax.set_xticks(x,['Новые','Старые']);ax.legend(frameon=False);finish(fig,ax,name,title,y=unit,note='До: 1 апреля–24 мая. После: 1 июня–31 августа.');chartdata(name,'bar',['Новые','Старые'],series,unit)
models=read('models');extra=read('extra_robustness');forest(pd.concat([models.iloc[:6],extra.iloc[[0,1,2]]]),'11_robustness','Эффект доли сохраняется при смене спецификации')
forest(models.iloc[[0,6,7,8,9,10]],'12_definitions','Проверки определения исхода и возраста')
forest(rr.sort_values('beta'),'13_regions','В регионах интервалы широкие',xlabel='Региональный DiD опасной доли, п.п.')
forest(read('leave_one_region_out'),'14_leaveout','Ни один регион не определяет общий результат','excluded')
ev=read('event_study');ev['label']=ev.month.map({5:'Май 1–24',6:'Июнь',7:'Июль',8:'Август'});forest(ev,'15_event','Разрыв возникает после мая','label')
pre=read('pretrends');pre['label']=['15–28 апреля','29 апреля–12 мая','13–24 мая'];forest(pre,'16_pretrends','Короткий предтренд не отвергает параллельность','label');manifest[-1]['note']=f"Совместный тест p={s['pretrend_p']:.3f}. Низкая мощность не доказывает параллельность."
fig,ax=plt.subplots(figsize=(10,8));d=reg.sort_values('liters_change_pct');ax.barh(d.region_name,d.liters_change_pct,color=BLUE);finish(fig,ax,'17_region_fuel','Региональный спад покупок топлива','Изменение литров на клиента / 30 дней, %',grid=False)
chartdata('17_region_fuel','barh',d.region_name,[dict(name='Изменение литров',values=d.liters_change_pct.tolist(),fill=BLUE)],'%')
fig,ax=plt.subplots(figsize=(11,6));ax.scatter(rr.shortage,rr.beta,s=50+rr.n/20,color=BLUE,alpha=.75)
for _,r in rr.iterrows():ax.annotate(r.model.replace(' область',' обл.'),(r.shortage,r.beta),xytext=(4,3),textcoords='offset points',fontsize=8)
ax.axhline(0,color=MUTED,ls='--');finish(fig,ax,'18_dose','Региональный градиент остаётся описательным','Стандартизированный индекс спада покупок','Региональный DiD опасной доли, п.п.',note='Индекс построен из post-данных. Он не является внешним инструментом дефицита.')
fig,ax=plt.subplots(figsize=(10,7.5));d=rr.sort_values('beta');vals=d[['new_change','old_change']].values;im=ax.imshow(vals,aspect='auto',cmap=cmap);ax.set_yticks(range(16),d.model);ax.set_xticks([0,1],['Новые','Старые']);fig.colorbar(im,ax=ax,label='Изменение доли, п.п.')
for i in range(16):
 for j in range(2):ax.text(j,i,f'{vals[i,j]:+.1f}',ha='center',va='center',color='white' if vals[i,j]>0 else INK,fontsize=10)
finish(fig,ax,'19_region_changes','Описательные изменения внутри каждого региона',grid=False)
fig,ax=plt.subplots(figsize=(10,5.5));b=ax.bar(prof.label,prof.median_price/1e6,color=[BLUE,SKY,NAVY]);ax.bar_label(b,fmt='%.2f',padding=4);finish(fig,ax,'20_car_price','Новизна автомобиля связана с его ценой',y='Медианная цена автомобиля, млн ₽',note='Цена автомобиля не равна доходу или богатству клиента.');chartdata('20_car_price','bar',prof.label,[dict(name='Медианная цена',values=(prof.median_price/1e6).tolist(),fill=BLUE)],'млн ₽')
corr=read('correlations').set_index('index');labels=['Возраст авто','Цена авто','Число авто','Штрафы 2025','Опасные до','Опасные после','Литры до','Литры после','Заправки до','Заправки после'];fig,ax=plt.subplots(figsize=(11,9));im=ax.imshow(corr,cmap=cmap,vmin=-1,vmax=1);ax.set_xticks(range(10),labels,rotation=50,ha='right');ax.set_yticks(range(10),labels);fig.colorbar(im,ax=ax,label='ρ Спирмена')
for i in range(10):
 for j in range(10):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if corr.iloc[i,j]>.45 else INK)
finish(fig,ax,'21_correlations','Связи признаков и клиентских исходов',grid=False)
mix=read('offence_mix');top=mix.groupby('offence_short_statement').n.sum().nlargest(6).index;short={'Превышение скорости на 20-40 км/ч':'Скорость 20–40','Не пристегнут ремень безопасности':'Ремень','Нарушение разметки':'Разметка','Превышение скорости на 40-60 км/ч':'Скорость 40–60','Использование телефона за рулем':'Телефон','Проезд на красный сигнал светофора':'Красный свет'}
fig,ax=plt.subplots(figsize=(11,6));x=np.arange(4);bottom=np.zeros(4);colors=[NAVY,BLUE,'#438BCE',SKY,'#A7D5F4','#D6ECFC','#F0F7FC'];series=[]
for i,key in enumerate(list(top)+['Остальные']):
 vals=[]
 for g,p in [('new','pre'),('new','post'),('old','pre'),('old','post')]:
  q=mix[(mix.group==g)&(mix.period==p)];vals.append(q.loc[q.offence_short_statement==key,'share'].sum() if key!='Остальные' else q.loc[~q.offence_short_statement.isin(top),'share'].sum())
 ax.bar(x,vals,bottom=bottom,color=colors[i],label=short.get(key,key));bottom+=vals;series.append(dict(name=short.get(key,key),values=vals,fill=colors[i]))
ax.set_xticks(x,['Новые до','Новые после','Старые до','Старые после']);ax.legend(bbox_to_anchor=(1.01,1),frameon=False,fontsize=10);finish(fig,ax,'22_mix','Состав зафиксированных нарушений',y='% штрафов');chartdata('22_mix','stacked',['Новые до','Новые после','Старые до','Старые после'],series,'%')
w=pd.read_pickle(ROOT/'tmp/clean_research_full/client_changes.pkl');fig,ax=plt.subplots(figsize=(11,5.5));bins=np.arange(-350,351,15)
for group,color,label in [('new',BLUE,'Новые'),('old',SKY,'Старые')]:ax.hist(w.loc[w.group==group,'delta_liters_30'].clip(-350,350),bins=bins,density=True,histtype='step',lw=2.5,color=color,label=label)
ax.legend(frameon=False);finish(fig,ax,'23_fuel_distribution','Распределение изменения покупок топлива','Изменение л / клиент / 30 дней','Плотность',note='Хвосты на графике ограничены ±350 л. Регрессия использует полные значения.')
# Analytic choropleth: real polygons, European/Siberian study extent, every sample region included.
geo=json.load(open(ROOT/'resources/geodata/candidates/russia_subjects_github.json'))
name_map={'Москва':'город федерального значения Москва','Санкт-Петербург':'город федерального значения Санкт-Петербург','Татарстан':'Республика Татарстан (Татарстан)','Башкортостан':'Республика Башкортостан'}
def mapplot(values,col,name,title,clabel,vmin=None,vmax=None):
 vals={name_map.get(r.region_name,r.region_name):getattr(r,col) for r in values.itertuples()};norm=Normalize(vmin if vmin is not None else min(vals.values()),vmax if vmax is not None else max(vals.values()));fig,ax=plt.subplots(figsize=(13,7));found=set()
 for ft in geo['features']:
  nm=ft['properties'].get('NL_NAME_1');val=vals.get(nm);g=ft['geometry'];polys=[g['coordinates']] if g['type']=='Polygon' else g['coordinates'];color=cmap(norm(val)) if val is not None else '#EDF1F4'
  if val is not None:found.add(nm)
  for poly in polys:
   coords=np.array(poly[0]);ax.add_patch(Polygon(coords,facecolor=color,edgecolor='white',lw=.35))
 # Capitals are tiny at this scale: labelled markers distinguish city from surrounding oblast.
 for nm,lon,lat in [('Москва',37.62,55.75),('Санкт-Петербург',30.32,59.94)]:
  val=vals[name_map[nm]];ax.scatter([lon],[lat],s=80,facecolor=cmap(norm(val)),edgecolor=NAVY,lw=1,zorder=5);ax.annotate(nm,(lon,lat),xytext=(-12,12),textcoords='offset points',fontsize=10,color=NAVY)
 ax.set_xlim(19,116);ax.set_ylim(43,73);ax.set_aspect(1.7);ax.axis('off');cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,fraction=.025,pad=.03);cb.set_label(clabel)
 finish(fig,ax,name,title,note='Серый — вне выборки. Показана часть России, содержащая все 16 регионов выборки.',grid=False)
 assert set(vals)==found,(set(vals)-found);return sorted(found)
found=mapplot(reg,'liters_change_pct','24_map_fuel','Карта изменения покупок топлива','Изменение литров, %')
mapplot(rr,'beta','25_map_effect','Карта региональных оценок','DiD опасной доли, п.п.')
(OUT/'map_validation.json').write_text(json.dumps({'matched':len(found),'expected':16,'names':found,'geometry':'resources/geodata/candidates/russia_subjects_github.json','note':'Границы используются как аналитическая подложка. Цвет вне выборки не означает нулевой эффект.'},ensure_ascii=False,indent=2))
# Plot mechanism on original units separately.
mech=read('mechanisms')
for key,name,title in [('liters_30','26_liters_effect','Дополнительное снижение литров не подтверждено'),('transactions_30','27_transactions_effect','Разница изменений числа заправок неточна')]:
 d=mech[mech.model==key].copy();d['label']=['Новые относительно старых'];forest(d,name,title,'label',d.unit.iloc[0])
# Monthly price and other graph values exported as native chart data.
(OUT/'chart_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(ROOT/'tmp/clean_research_full/native_charts.json').write_text(json.dumps(native,ensure_ascii=False));print('Generated',len(manifest),'charts')
