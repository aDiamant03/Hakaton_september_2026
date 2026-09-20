# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'presentation_charts_v2_20260920'; OUT.mkdir(parents=True,exist_ok=True)
NAVY='#123B7A'; BLUE='#2563EB'; SKY='#38BDF8'; PALE='#DBEAFE'; INK='#132238'; MUTED='#5B6B82'; GRID='#DCE5F0'; RED='#E25555'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':22,'axes.labelsize':13,'figure.facecolor':'white','axes.facecolor':'white'})

def base(title,subtitle=''):
    fig,ax=plt.subplots(figsize=(12.8,7.2));
    fig.suptitle(title,x=.07,y=.96,ha='left',fontsize=23,fontweight='bold',color=INK)
    if subtitle: fig.text(.07,.895,subtitle,ha='left',fontsize=12,color=MUTED)
    ax.spines[['top','right']].set_visible(False); ax.spines[['left','bottom']].set_color(GRID); ax.tick_params(colors=MUTED)
    ax.grid(axis='y',color=GRID,lw=.8); ax.set_axisbelow(True); return fig,ax
def save(fig,name,bottom=.10): fig.tight_layout(rect=[.06,bottom,.98,.86]); fig.savefig(OUT/name,dpi=190,bbox_inches='tight'); plt.close(fig)
def pct(x): return f'{x:+.0f}%'.replace('.',',')

def main():
    TD=ROOT/'outputs/wealth_newness_20260919/theory_charts'
    pre=pd.read_csv(TD/'chart_data_prepost.csv'); ch=pd.read_csv(TD/'chart_data_changes.csv'); reg=pd.read_csv(TD/'chart_data_regions.csv')
    shortage=pd.read_csv(ROOT/'outputs/wealth_newness_20260919/physical_shortage_regions.csv')
    sens=pd.read_csv(ROOT/'outputs/wealth_newness_20260919/new_vs_old_sensitivity.csv'); sens=sens[sens.outcome.eq('high_risk')].sort_values('new_age_cutoff')
    summ=json.load(open(ROOT/'outputs/fuel_cleaning_20260919/cleaning_summary.json'))
    f=pd.read_csv(ROOT/'fuel_transaction.csv',sep=';',decimal=','); f['date']=pd.to_datetime(f.order_datetime); f['month']=f.date.dt.month
    c=pd.read_csv(ROOT/'clients_demographics.csv',sep=';',decimal=','); c['signup']=pd.to_datetime(c.subscription_creation_date)

    # 1. Cleaning funnel
    d=pd.DataFrame(summ['counts']); labels=['Клиенты','Штрафы','Заправки']; before=d.before.to_numpy(); after=d.after.to_numpy()
    fig,ax=base('Очистка убрала несопоставимые наблюдения','Главная причина — регистрация клиента после начала исторического окна')
    y=np.arange(3); ax.barh(y,before,color=PALE,height=.55,label='До очистки'); ax.barh(y,after,color=BLUE,height=.55,label='После очистки')
    xmax=max(before.max(),after.max())
    for i,(b,a) in enumerate(zip(before,after)):
        # Разносим подписи по вертикали: на коротких полосах текст не должен
        # сливаться с подписью второй серии.
        ax.text(b,i-.12,f'{b/1000:.1f} тыс.',va='center',ha='right',color=MUTED,fontweight='bold')
        if a > .12*xmax:
            ax.text(a*.985,i+.12,f'{a/1000:.1f} тыс.',va='center',ha='right',color='white',fontweight='bold')
        else:
            ax.text(a+xmax*.012,i+.12,f'{a/1000:.1f} тыс.',va='center',ha='left',color=NAVY,fontweight='bold')
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:.0f} тыс.')); ax.legend(frameon=False,loc='lower right'); ax.set_xlabel('Число строк'); save(fig,'01_очистка_данных.png')

    # 2. Outlier audit, four compact panels
    fig,axs=plt.subplots(2,2,figsize=(12.8,7.2)); fig.suptitle('Выбросы выявлены до моделирования',x=.07,y=.97,ha='left',fontsize=23,fontweight='bold',color=INK)
    clean_paths=[ROOT/'outputs/fuel_cleaning_20260919/fuel_transaction_clean.csv',ROOT/'outputs/fuel_cleaning_20260919/clients_demographics_clean.csv',ROOT/'outputs/fuel_cleaning_20260919/fines_2026_clean.csv']
    fc=pd.read_csv(clean_paths[0],sep=';',decimal=','); cc=pd.read_csv(clean_paths[1],sep=';',decimal=','); yc=pd.read_csv(clean_paths[2],sep=';'); yraw=pd.read_csv(ROOT/'fines_2026.csv',sep=';')
    specs=[(f,fc,'order_fuel_volume','Объём заправки, л'),(f,fc,'order_fuel_price_1liter','Цена литра, руб.'),(c,cc,'price','Стоимость авто, руб.'),(yraw,yc,'total_fine_amount','Сумма штрафа, коп.')]
    for ax,(raw,cln,col,title) in zip(axs.flat,specs):
        vals=[pd.to_numeric(raw[col],errors='coerce').dropna(),pd.to_numeric(cln[col],errors='coerce').dropna()]
        ax.boxplot(vals,tick_labels=['До','После'],showfliers=False,patch_artist=True,boxprops={'facecolor':SKY,'alpha':.7},medianprops={'color':NAVY,'linewidth':2}); ax.set_title(title,loc='left',fontweight='bold',color=INK); ax.grid(axis='y',color=GRID)
        q1,q3=vals[0].quantile([.25,.75]); lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1); n=((vals[0]<lo)|(vals[0]>hi)).sum(); ax.text(.98,.92,f'Выбросов по IQR: {n:,}'.replace(',',' '),transform=ax.transAxes,ha='right',va='top',color=MUTED,fontsize=10)
    fig.tight_layout(rect=[.05,.05,.98,.90]); fig.savefig(OUT/'02_аудит_выбросов.png',dpi=190,bbox_inches='tight'); plt.close(fig)

    # 3. Indexed fuel dynamics
    m=f[f.month.between(4,8)].groupby('month').agg(liters=('order_fuel_volume','sum'),orders=('order_id','size')); idx=m/m.iloc[0]*100
    fig,ax=base('После мая физическая доступность топлива падает','Оба показателя приведены к апрелю = 100, поэтому двойная ось не нужна')
    months=['Апрель','Май','Июнь','Июль','Август']; ax.plot(months,idx.liters,lw=3.5,marker='o',ms=9,color=BLUE,label='Литры'); ax.plot(months,idx.orders,lw=3.5,marker='s',ms=8,color=SKY,label='Число заправок'); ax.axhline(100,color=GRID,lw=1); ax.fill_between(range(1,5),idx[['liters','orders']].min(axis=1).iloc[1:],100,color=PALE,alpha=.35)
    for i,v in enumerate(idx.liters): ax.text(i,v+2,f'{v:.0f}',ha='center',color=BLUE,fontweight='bold');
    ax.set_ylabel('Индекс, апрель = 100'); ax.legend(frameon=False,ncol=2,loc='lower left'); save(fig,'03_динамика_дефицита.png')

    # 4. Shortage lollipop
    s=shortage.sort_values('physical_shortage'); fig,ax=base('Дефицит был региональным, а не общим','Положительный индекс означает более сильное падение литров и частоты заправок')
    yy=np.arange(len(s)); ax.hlines(yy,0,s.physical_shortage,color=PALE,lw=5); ax.scatter(s.physical_shortage,yy,c=np.where(s.physical_shortage>0,BLUE,SKY),s=90,zorder=3); ax.axvline(0,color=MUTED,lw=1); ax.set_yticks(yy,s.region_name); ax.set_xlabel('Индекс физического дефицита, стандартные отклонения'); ax.grid(False); save(fig,'04_дефицит_по_регионам.png')

    # 5. Vehicle age segmentation
    v=c.sort_values(['auto_document_id','signup']).drop_duplicates('auto_document_id'); p=v.groupby('client_id').auto_year.median(); age=(2026-p).clip(0,30)
    fig,ax=base('Возраст автомобиля естественно делит клиентов на три группы','Основное сравнение использует контрастные группы: до 5 лет и от 11 лет')
    ax.hist(age,bins=np.arange(0,31),color=SKY,edgecolor='white'); ax.axvspan(0,5,color=PALE,alpha=.8,label='Новые: 0–5'); ax.axvspan(11,30,color=BLUE,alpha=.12,label='Старые: 11+'); ax.axvline(5,color=BLUE,ls='--'); ax.axvline(10,color=NAVY,ls='--'); ax.set_xlabel('Возраст автомобиля, лет'); ax.set_ylabel('Клиенты'); ax.legend(frameon=False); save(fig,'05_сегментация_по_возрасту.png')

    # 6. Main slope chart
    fig,ax=base('После дефицита новые и старые автомобили расходятся','Доля опасных нарушений снижается у новых и растёт у старых')
    periods=['До дефицита','После дефицита']; colors={'Новые, до 5 лет':SKY,'Старые, от 11 лет':NAVY}
    for group,d in pre.groupby('vehicle_group'):
        d=d.set_index('period').loc[['До дефицита','После начала дефицита']]; vals=d.dangerous_share_pct.to_numpy(); ax.plot([0,1],vals,lw=4,marker='o',ms=11,color=colors[group]);
        for x,z in enumerate(vals): ax.text(x,z+.08,f'{z:.2f}%'.replace('.',','),ha='center',color=colors[group],fontweight='bold',fontsize=15)
        ax.text(1.03,vals[1],group,va='center',color=colors[group],fontweight='bold')
    ax.set_xticks([0,1],periods); ax.set_xlim(-.12,1.38); ax.set_ylim(4.9,7.05); ax.set_ylabel('Опасные нарушения среди всех штрафов, %'); ax.grid(axis='y',color=GRID); save(fig,'06_главный_результат.png')

    # 7. Diverging rate bars
    d=ch.set_index('vehicle_group').loc[['Новые, до 5 лет','Старые, от 11 лет']]; fig,ax=base('Опасные нарушения сокращаются только у новых автомобилей','Частота рассчитана на 1 000 клиентов за 30 дней')
    y=np.arange(2); h=.28; danger=d.dangerous_rate_pct_change.to_numpy(); light=d.light_rate_pct_change.to_numpy(); ax.barh(y+h/2,danger,h,color=NAVY,label='Опасные'); ax.barh(y-h/2,light,h,color=SKY,label='Лёгкие'); ax.axvline(0,color=MUTED,lw=1)
    for yy,z in zip(y+h/2,danger): ax.text(z+(.5 if z>=0 else -.5),yy,pct(z),ha='left' if z>=0 else 'right',va='center',fontweight='bold',color=NAVY)
    for yy,z in zip(y-h/2,light): ax.text(z+.5,yy,pct(z),va='center',fontweight='bold',color=SKY)
    ax.set_yticks(y,['Новые авто','Старые авто']); ax.invert_yaxis(); ax.set_xlabel('Изменение частоты, %'); ax.legend(frameon=False,ncol=2); save(fig,'07_опасные_и_легкие.png')

    # 8. Regional dose response
    fig,ax=base('Чем сильнее дефицит, тем больше разрыв между старыми и новыми','16 регионов; положительный разрыв означает более сильное ухудшение старых автомобилей')
    x=reg.physical_shortage.to_numpy(); yy=reg.old_minus_new_gap_pp.to_numpy(); ax.scatter(x,yy,s=95,color=BLUE,edgecolor='white',lw=1.2,zorder=3); coef=np.polyfit(x,yy,1); xx=np.linspace(x.min(),x.max(),100); ax.plot(xx,np.polyval(coef,xx),color=SKY,lw=3); ax.axhline(0,color=GRID,lw=1)
    for _,z in reg.iterrows(): ax.annotate(str(z.region_name).replace(' область',''),(z.physical_shortage,z.old_minus_new_gap_pp),xytext=(5,4),textcoords='offset points',fontsize=8,color=MUTED)
    ax.text(.03,.93,'r = 0,82   p = 0,0001',transform=ax.transAxes,color=NAVY,fontweight='bold',fontsize=16); ax.set_xlabel('Индекс физического дефицита'); ax.set_ylabel('Разрыв старые − новые, п.п.'); save(fig,'08_региональная_зависимость.png')

    # 9. Forest robustness
    fig,ax=base('Вывод не зависит от выбранной границы «нового» автомобиля','OR < 1 означает меньший рост риска у новых автомобилей относительно старых')
    y=np.arange(len(sens)); ax.errorbar(sens.estimate,y,xerr=[sens.estimate-sens.ci_low,sens.ci_high-sens.estimate],fmt='o',ms=9,color=NAVY,ecolor=SKY,capsize=6,lw=2); ax.axvline(1,color=MUTED,ls='--'); ax.set_yticks(y,[f'Новые ≤ {v} лет' for v in sens.new_age_cutoff]); ax.invert_yaxis(); ax.set_xlabel('Отношение шансов опасного нарушения, OR'); ax.grid(axis='x',color=GRID); ax.grid(axis='y',visible=False); save(fig,'09_устойчивость_результата.png')

    pd.DataFrame([
        ('Очистка','01_очистка_данных.png'),('Очистка','02_аудит_выбросов.png'),('Преданализ','03_динамика_дефицита.png'),('Преданализ','04_дефицит_по_регионам.png'),('Преданализ','05_сегментация_по_возрасту.png'),('Гипотеза','06_главный_результат.png'),('Гипотеза','07_опасные_и_легкие.png'),('Гипотеза','08_региональная_зависимость.png'),('Устойчивость','09_устойчивость_результата.png')],columns=['Раздел','Файл']).to_csv(OUT/'порядок_графиков.csv',index=False)
    print('created 9 charts')
if __name__=='__main__': main()
