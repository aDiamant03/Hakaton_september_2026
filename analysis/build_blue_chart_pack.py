# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'blue_chart_pack_20260920'; OUT.mkdir(parents=True,exist_ok=True)
BLUES=['#dbeafe','#93c5fd','#60a5fa','#3b82f6','#1d4ed8','#0f3d8a']; CYAN='#06b6d4'; GRID='#d1d5db'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':15,'axes.labelsize':11,'figure.dpi':130})

def finish(fig,ax,title,name,xlabel='',ylabel=''):
    ax.set_title(title,loc='left',fontweight='bold'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(axis='y',alpha=.35,color=GRID); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT/name,dpi=180,bbox_inches='tight'); plt.close(fig)

def main():
    c=pd.read_csv(ROOT/'clients_demographics.csv',sep=';',decimal=','); f=pd.read_csv(ROOT/'fuel_transaction.csv',sep=';',decimal=','); y=pd.read_csv(ROOT/'fines_2026.csv',sep=';')
    cc=pd.read_csv(ROOT/'outputs/fuel_cleaning_20260919/clients_demographics_clean.csv',sep=';',decimal=',')
    fc=pd.read_csv(ROOT/'outputs/fuel_cleaning_20260919/fuel_transaction_clean.csv',sep=';',decimal=',')
    yc=pd.read_csv(ROOT/'outputs/fuel_cleaning_20260919/fines_2026_clean.csv',sep=';')
    summary=json.load(open(ROOT/'outputs/fuel_cleaning_20260919/cleaning_summary.json'))
    manifest=[]

    # 01 cleaning volume
    d=pd.DataFrame(summary['counts']); fig,ax=plt.subplots(figsize=(10,5)); d.set_index('dataset')[['before','after']].plot.bar(ax=ax,color=[BLUES[1],BLUES[4]],rot=0)
    ax.legend(['До очистки','После очистки']); finish(fig,ax,'Сколько строк осталось после очистки','01_строки_до_и_после_очистки.png',ylabel='Число строк'); manifest.append(('Очистка','01_строки_до_и_после_очистки.png','Масштаб фильтрации'))

    # 02 reasons removed
    r=d.set_index('dataset')[['registration','electrified_after_registration','low_price_after_client_filters','unexplained_negative_after_other_filters']]
    fig,ax=plt.subplots(figsize=(11,5)); r.plot.bar(stacked=True,ax=ax,color=BLUES[1:5],rot=0); ax.legend(['Регистрация после cutoff','Электромобили','Цена топлива <50','Необъяснённый минус'],fontsize=9)
    finish(fig,ax,'Почему удалялись строки','02_причины_исключения.png',ylabel='Удалённые строки'); manifest.append(('Очистка','02_причины_исключения.png','Причины фильтрации'))

    # 03-06 boxplots
    pairs=[('order_fuel_volume','Объём заправки, л',f,fc,'03_выбросы_объема_топлива.png'),('order_fuel_price_1liter','Цена литра, руб.',f,fc,'04_выбросы_цены_топлива.png'),('price','Стоимость автомобиля, руб.',c,cc,'05_выбросы_стоимости_авто.png'),('total_fine_amount','Сумма штрафа, коп.',y,yc,'06_выбросы_штрафов.png')]
    for col,label,b,a,name in pairs:
        vals=[pd.to_numeric(b[col],errors='coerce').dropna(),pd.to_numeric(a[col],errors='coerce').dropna()]
        fig,ax=plt.subplots(figsize=(9,4.8)); ax.boxplot(vals,labels=['До очистки','После очистки'],patch_artist=True,showfliers=True,boxprops={'facecolor':BLUES[2]},medianprops={'color':BLUES[5],'linewidth':2},flierprops={'marker':'.','markerfacecolor':CYAN,'alpha':.25})
        finish(fig,ax,f'Выбросы: {label.lower()}',name,ylabel=label); manifest.append(('Очистка',name,'Выбросы до и после'))

    # client-level vehicle table
    c['signup']=pd.to_datetime(c['subscription_creation_date']); v=c.sort_values(['auto_document_id','signup']).drop_duplicates('auto_document_id')
    p=v.groupby('client_id').agg(region=('kladr_code','first'),auto_year=('auto_year','median'),price=('price','median'),cars=('auto_document_id','nunique'))
    p['vehicle_age']=2026-p.auto_year; p['vehicle_group']=pd.cut(p.vehicle_age,[-1,5,10,100],labels=['Новые 0–5','Средние 6–10','Старые 11+'])

    # 07 age distribution
    fig,ax=plt.subplots(figsize=(10,5)); p.vehicle_age.clip(0,35).plot.hist(bins=35,ax=ax,color=BLUES[3],edgecolor='white'); ax.axvline(5,color=CYAN,ls='--'); ax.axvline(10,color=BLUES[5],ls='--')
    finish(fig,ax,'Распределение возраста автомобилей','07_возраст_автомобилей.png','Возраст автомобиля, лет','Клиенты'); manifest.append(('Преданализ','07_возраст_автомобилей.png','Выбор границ новых и старых авто'))

    # 08 group size and median price
    gp=p.groupby('vehicle_group',observed=True).agg(clients=('region','size'),median_price=('price','median')).reset_index()
    fig,ax=plt.subplots(figsize=(9,5)); ax.bar(gp.vehicle_group.astype(str),gp.clients,color=BLUES[2:5]);
    for i,row in gp.iterrows(): ax.text(i,row.clients,f"{int(row.clients):,}\nмедиана {row.median_price/1e6:.1f} млн",ha='center',va='bottom')
    finish(fig,ax,'Размер групп и медианная стоимость автомобиля','08_группы_авто.png',ylabel='Число клиентов'); manifest.append(('Преданализ','08_группы_авто.png','Размер и различия групп'))

    # 09 regions clients
    names={77:'Москва',50:'Московская обл.',78:'Санкт-Петербург',16:'Татарстан',66:'Свердловская обл.',54:'Новосибирская обл.',47:'Ленинградская обл.',52:'Нижегородская обл.',23:'Краснодарский край',63:'Самарская обл.',74:'Челябинская обл.',42:'Кемеровская обл.',72:'Тюменская обл.',2:'Башкортостан',24:'Красноярский край',55:'Омская обл.'}
    rg=p.region.map(names).value_counts().sort_values(); fig,ax=plt.subplots(figsize=(9,7)); rg.plot.barh(ax=ax,color=BLUES[3]); finish(fig,ax,'Клиенты по регионам','09_клиенты_по_регионам.png','Число клиентов'); manifest.append(('Преданализ','09_клиенты_по_регионам.png','Региональная структура выборки'))

    # 10 monthly fuel
    f['date']=pd.to_datetime(f.order_datetime); f['month']=f.date.dt.month
    m=f[f.month.between(4,8)].groupby('month').agg(liters=('order_fuel_volume','sum'),orders=('order_id','size'))
    fig,ax=plt.subplots(figsize=(10,5)); ax.plot(m.index,m.liters/1e6,marker='o',lw=2.5,color=BLUES[4],label='Литры, млн'); ax2=ax.twinx(); ax2.plot(m.index,m.orders/1000,marker='s',lw=2.5,color=CYAN,label='Заправки, тыс.'); ax.set_xticks(m.index,['Апрель','Май','Июнь','Июль','Август']); ax2.set_ylabel('Заправки, тыс.'); ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    finish(fig,ax,'После мая снижаются литры и число заправок','10_динамика_топлива.png',ylabel='Объём топлива, млн л'); manifest.append(('Путь к гипотезе','10_динамика_топлива.png','Начало физического дефицита'))

    # 11 shortage regions
    sr=pd.read_csv(ROOT/'outputs/wealth_newness_20260919/physical_shortage_regions.csv').sort_values('physical_shortage'); fig,ax=plt.subplots(figsize=(9,7)); ax.barh(sr.region_name,sr.physical_shortage,color=[BLUES[1] if z<0 else BLUES[4] for z in sr.physical_shortage]); ax.axvline(0,color='black',lw=.8)
    finish(fig,ax,'Физический дефицит сильно различается по регионам','11_дефицит_по_регионам.png','Индекс дефицита, стандартные отклонения'); manifest.append(('Путь к гипотезе','11_дефицит_по_регионам.png','Региональная интенсивность шока'))

    # Load prepared hypothesis data
    TD=ROOT/'outputs/wealth_newness_20260919/theory_charts'; pre=pd.read_csv(TD/'chart_data_prepost.csv'); ch=pd.read_csv(TD/'chart_data_changes.csv'); reg=pd.read_csv(TD/'chart_data_regions.csv')
    # 12 slope
    fig,ax=plt.subplots(figsize=(9,5)); periods=['До дефицита','После начала дефицита']; colors=[BLUES[2],BLUES[5]]
    for color,(group,dg) in zip(colors,pre.groupby('vehicle_group')):
        dg=dg.set_index('period').loc[periods]; ax.plot([0,1],dg.dangerous_share_pct,marker='o',lw=3,label=group,color=color)
        for x,val in enumerate(dg.dangerous_share_pct): ax.text(x,val+.08,f'{val:.2f}%',ha='center',color=color,fontweight='bold')
    ax.set_xticks([0,1],periods); ax.set_ylim(pre.dangerous_share_pct.min()-.15,pre.dangerous_share_pct.max()+.35); ax.legend(); finish(fig,ax,'Доля опасных нарушений: новые против старых','12_опасная_доля_до_после.png',ylabel='Доля опасных штрафов, %'); manifest.append(('Подтверждение','12_опасная_доля_до_после.png','Главный результат'))

    # 13 rate change
    d=ch.set_index('vehicle_group')[['dangerous_rate_pct_change','light_rate_pct_change']]; d.columns=['Опасные','Лёгкие']; fig,ax=plt.subplots(figsize=(9,5)); d.plot.bar(ax=ax,color=[BLUES[5],BLUES[2]],rot=0); ax.axhline(0,color='black',lw=.8); ax.legend()
    finish(fig,ax,'Изменение частоты штрафов на 1 000 клиентов','13_частота_опасных_и_легких.png',ylabel='Изменение, %'); manifest.append(('Подтверждение','13_частота_опасных_и_легких.png','Нормированная частота'))

    # 14 regional scatter
    fig,ax=plt.subplots(figsize=(9,6)); ax.scatter(reg.physical_shortage,reg.old_minus_new_gap_pp,s=65,color=BLUES[4],alpha=.85); coef=np.polyfit(reg.physical_shortage,reg.old_minus_new_gap_pp,1); xx=np.linspace(reg.physical_shortage.min(),reg.physical_shortage.max(),100); ax.plot(xx,np.polyval(coef,xx),color=CYAN,lw=2)
    for _,z in reg.iterrows(): ax.annotate(str(z.region_name).replace(' область',''),(z.physical_shortage,z.old_minus_new_gap_pp),xytext=(4,3),textcoords='offset points',fontsize=8)
    finish(fig,ax,'Чем сильнее дефицит, тем больше разрыв старые − новые','14_региональная_дозозависимость.png','Индекс физического дефицита','Разрыв в изменении опасной доли, п.п.'); manifest.append(('Подтверждение','14_региональная_дозозависимость.png','Дозозависимость по регионам'))

    # 15 heatmap
    hm=reg.set_index('region_name')[['new_change_pp','old_change_pp']].sort_values('physical_shortage' if 'physical_shortage' in reg.columns else 'new_change_pp') if False else reg.sort_values('physical_shortage',ascending=False).set_index('region_name')[['new_change_pp','old_change_pp']]
    fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(hm.values,aspect='auto',cmap='Blues'); ax.set_xticks([0,1],['Новые авто','Старые авто']); ax.set_yticks(range(len(hm)),hm.index); plt.colorbar(im,ax=ax,label='Изменение опасной доли, п.п.')
    for i in range(len(hm)):
        for j in range(2): ax.text(j,i,f'{hm.iloc[i,j]:+.1f}',ha='center',va='center',color='white' if hm.iloc[i,j]>2 else 'black',fontsize=8)
    finish(fig,ax,'Как меняется опасная доля в каждом регионе','15_тепловая_карта_регионов.png'); manifest.append(('Подтверждение','15_тепловая_карта_регионов.png','Региональная неоднородность'))

    # 16 forest sensitivity
    sens=pd.read_csv(ROOT/'outputs/wealth_newness_20260919/new_vs_old_sensitivity.csv'); sens=sens[sens.outcome.eq('high_risk')].sort_values('new_age_cutoff')
    fig,ax=plt.subplots(figsize=(8,4.5)); yy=np.arange(len(sens)); ax.errorbar(sens.estimate,yy,xerr=[sens.estimate-sens.ci_low,sens.ci_high-sens.estimate],fmt='o',color=BLUES[5],ecolor=BLUES[2],capsize=5); ax.axvline(1,color='black',ls='--'); ax.set_yticks(yy,[f'Новые ≤{x} лет' for x in sens.new_age_cutoff]); ax.invert_yaxis()
    finish(fig,ax,'Результат устойчив к границе «нового» автомобиля','16_устойчивость_границ_новизны.png','Отношение шансов опасного нарушения (OR)'); manifest.append(('Устойчивость','16_устойчивость_границ_новизны.png','Порог 3, 5 и 7 лет'))

    # 17 top offence categories pre/post
    y['date']=pd.to_datetime(y.bill_offence_date); y['period']=np.where(y.date<pd.Timestamp('2026-06-01'),'До 1 июня','После 1 июня'); top=y.offence_short_statement.value_counts().head(8).index; z=y[y.offence_short_statement.isin(top)].groupby(['offence_short_statement','period']).size().unstack(fill_value=0); z=z.div(z.sum(axis=0),axis=1)*100
    fig,ax=plt.subplots(figsize=(10,6)); z.sort_values('После 1 июня').plot.barh(ax=ax,color=[BLUES[2],BLUES[5]]); ax.legend(); finish(fig,ax,'Структура наиболее частых нарушений до и после 1 июня','17_структура_типов_нарушений.png','Доля среди показанных категорий, %'); manifest.append(('Преданализ','17_структура_типов_нарушений.png','Изменение состава штрафов'))

    pd.DataFrame(manifest,columns=['section','file','purpose']).to_csv(OUT/'manifest.csv',index=False)
    print(f'created {len(manifest)} charts in {OUT}')

if __name__=='__main__': main()
