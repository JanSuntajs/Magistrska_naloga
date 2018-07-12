import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.patches as patches
# import networkx as nx
import os
plt.rc('text', usetex = False)
# plt.rc('font',  family = 'sans-serif')
plt.rc('text', usetex = True)
plt.rc('font',family='serif',serif=['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

#show disorder
def plot_disorder():
    # fig, (ax1,ax2,ax3)=plt.subplots(1,3, figsize=(18,6), subplot_kw=dict(aspect=1,adjustable='box'))
    fig, ((ax1,ax2), (ax3, ax4))=plt.subplots(2,2,figsize=(14,14))
    # ax1=plt.subplot2grid((2,2), (0,0))
    # ax2=plt.subplot2grid((2,2), (0,1))
    # ax3=plt.subplot2grid((2,2), (1,0),rowspan=2)
    # names=['a)','b)', 'c)']




    for i,ax in enumerate((ax1,ax2,ax3,ax4)):
        ax.set_xlim(-0.8,4.8)
        ax.set_ylim(-0.8,4.8    )
        # ax.set_xlabel(names[i])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # cur_axes = plt.gca()
        # cur_axes.axes.get_xaxis().set_visible(False)
        # cur_axes.axes.get_yaxis().set_visible(False)

    lattice=np.array([np.array([i,j]) for j in range(5) for i in range(5)])
    print(lattice)
    connections=np.array([[el1, el2] for el1 in lattice for el2 in lattice if np.linalg.norm(el2-el1)==1.])

    # G=nx.grid_2d_graph(4,4)
    # pos=dict((n,n) for n in G.nodes())
    # nx.draw_networkx(G, pos=pos, ax=ax4)


    for site in lattice:
        circle1=plt.Circle(site, 0.2, color='k')
        circle2=plt.Circle(site,np.random.uniform(0.1,0.3),color='k')
        circle3=plt.Circle(site+np.random.uniform(-0.5,0.5,2),0.2,color='k')
        ax1.add_artist(circle1)
        ax2.add_artist(circle2)
        ax3.add_artist(circle3)

    # for i in range(5):
    #   ax4.axvline(x=i,ymin=0.8/5.6,ymax=4.8/5.6,color='k',lw=1)
    #   ax4.axhline(y=i,xmin=0.8/5.6,xmax=4.8/5.6, color='k',lw=1)
    for connection in connections:
        ax4.plot(connection.T[0], connection.T[1],lw=np.random.uniform(1,10),color='k' )
    plt.tight_layout()
    plt.savefig('disorder_scheme.pdf')



def plot_DOS(alpha=1.1,x0=1,xmax=5,scale=1):

    engy1=np.linspace(-xmax,-x0,300)
    engy2=np.linspace(-x0,x0,300)
    engy3=np.linspace(x0,xmax,300)
    beta=-alpha*np.tan(alpha*x0)


    dos1=scale*np.exp(-beta*(engy1+x0))*np.cos(alpha*x0)

    dos2=scale*np.cos(alpha*engy2)

    dos3=np.exp(beta*(engy3-x0))*np.cos(alpha*x0)*scale


    fig,ax=plt.subplots(figsize=(12,6))
    # ax.set_aspect('equal')
    # ax.axis('off')
    ax.set_frame_on(False)
    ax.fill_between(engy1,0,dos1,color='red',alpha=0.3)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.text(-1.5,-0.14,'$E_c$', fontsize=42)
    ax.text(1,-0.14,'$E\'_c$', fontsize=42)
    ax.text(-5,0.9,'\\textbf{DOS}', fontsize=44)
    plt.fill_between(engy2,0,dos2,color='blue',alpha=0.3)
    plt.fill_between(engy3,0,dos3, color='red',alpha=0.3)
    plt.tight_layout(pad=3)
    plt.savefig('mobility_edge_DOS.pdf')



def plot_loc_ext():

    # engy1=np.linspace(-xmax,-x0,300)
    # engy2=np.linspace(-x0,x0,300)
    # engy3=np.linspace(x0,xmax,300)
    # beta=-alpha*np.tan(alpha*x0)


    # dos1=scale*np.exp(-beta*(engy1+x0))*np.cos(alpha*x0)

    # dos2=scale*np.cos(alpha*engy2)

    # dos3=np.exp(beta*(engy3-x0))*np.cos(alpha*x0)*scale


    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6,6))
    
    # ax.axis('off')
    for ax in (ax1,ax2):
        # ax.set_aspect('equal')
        ax.set_frame_on(False)
        ax.set_ylim(-2,2)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        # ax.text(-1.5,-0.14,'$E_c$', fontsize=42)
        # ax.text(1,-0.14,'$E\'_c$', fontsize=42)
        # ax.text(-5,0.9,'\\textbf{DOS}', fontsize=44)

    x=np.linspace(-30,30,10000)
    y1=np.cos(0.9*x)+0.3*np.cos(0.7*x)+0.8*np.cos(1.8*x)
    y1=y1/2.1
    ax1.plot(x,y1)
    ax1.set_ylim(-1.4,1.1)
    ax2.set_ylim(-1.4,1.1)
    y2=(np.cos(1.2*x)+np.cos(0.8*x))/2
    ax1.axhline(y=0,color='black')
    ax2.axhline(y=0, color='black')
    ax2.plot(x,y2*np.exp(-np.abs(x)/5))
    ax2.plot(x,-np.exp(-np.abs(x)/5), color='black', ls='--')
    ax2.plot(x,np.exp(-np.abs(x)/5),color='black',ls='--')
    ax1.text(-0.25,-1.25,'\\textbf{a)}', fontsize=26)
    ax2.text(-0.2,-1.25,'\\textbf{b)}', fontsize=26)
    # ax1.text(-25,-1.25,'\\textbf{ Extended}', fontsize=36)
    # ax2.text(-23,-1.25,'\\textbf{ Localized}', fontsize=36)
    plt.tight_layout()
    # plt.show()
    cwd=os.getcwd()
    # os.chdir('Presentation')
    plt.savefig('diff_loc_ext1.pdf')
    # os.chdir(cwd)


def plot_box(W=3,a=2):

    fontsize=[32,34,36]
    fig,ax=plt.subplots(figsize=(12,6))
    # ax.set_aspect('equal')
    # ax.axis('off')
    xticks=[-W,0,W]
    xlabels1=['$-W$','$0$', '$W$']
    xlabels2=['$-H$','$0$', '$H$']
    yticks=[0,1/(2*W)]
    ylabels1=['$0$','$\\frac{1}{2W}$']
    ylabels2=['$0$','$\\frac{1}{2H}$']
    ax.axhline(y=0, xmin=-W-a-0.1, xmax=W+a+0.1)
    


    patch=patches.Rectangle((-W,0), 2*W, 1/(2*W))
    # ax.tick_params(axis='x', labelsize=fontsize[1])
    # ax.tick_params(axis='y', labelsize=fontsize[1])
    # ax.tick_params(axis='x', labelsize=fontsize[1],pad=5,direction='out')
    ax.add_patch(patch)

    ax1=ax.twiny()
    ax2=ax.twinx()

    ax.set_xlabel('$w_j$',fontsize=fontsize[-1],color='blue')
    ax.set_ylabel('$p(w_j)$',fontsize=fontsize[-1],color='blue')
    ax2.set_ylabel('$p(h_j)$',fontsize=fontsize[-1], color='red')
    ax1.set_xlabel('$h_j$',fontsize=fontsize[-1], color='red')
    for ax_ in (ax,ax1,ax2):

        ax_.tick_params(axis='x', labelsize=fontsize[1])
        ax_.tick_params(axis='y', labelsize=fontsize[1])
        ax_.tick_params(axis='x', labelsize=fontsize[1],pad=1,direction='out')

    for ax_ in (ax,ax1):

        ax_.set_xticks(xticks)
        ax_.set_xlim(-W-a,W+a)
    
    for ax_ in (ax,ax2):

        ax_.set_yticks(yticks)
        ax_.set_ylim(-0.01,1/(2*W) +0.05)

    
    # ax1.add_patch(patch)
    ax.set_xticklabels(xlabels1, color='blue')
    ax.set_yticklabels(ylabels1, color='blue')

    ax1.set_xticklabels(xlabels2, color='red')
    ax2.set_yticklabels(ylabels2, color='red')
        
    
    
    plt.tight_layout(pad=1)
    cwd=os.getcwd()
    # os.chdir('Presentation')
    plt.savefig('prob_dist.pdf')
    os.chdir(cwd)
    # plt.show()

    def plot_ipr_example(W=3,a=2):

            fontsize=[32,34,36]
        fig,ax=plt.subplots(figsize=(12,6))
        # ax.set_aspect('equal')
        # ax.axis('off')
        xticks=[-W,0,W]
        xlabels=['$-W$','$0$', '$W$']
        yticks=[0,1/(2*W)]
        ylabels=['$0$','$\\frac{1}{2W}$']
        ax.axhline(y=0, xmin=-W-a-0.1, xmax=W+a+0.1)
        ax.set_xlim(-W-a,W+a)
        ax.set_ylim(-0.01,1/(2*W) +0.05)
        ax.set_ylabel('$p(\\varepsilon_j)$',fontsize=fontsize[-1])
        ax.set_xlabel('$\\varepsilon_j$',fontsize=fontsize[-1])
        patch=patches.Rectangle((-W,0), 2*W, 1/(2*W))
        ax.tick_params(axis='x', labelsize=fontsize[1])
        ax.tick_params(axis='y', labelsize=fontsize[1])
        ax.tick_params(axis='x', labelsize=fontsize[1],pad=5,direction='out')
        ax.add_patch(patch)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        # ax.spines['left'].set_position('zero')
        # ax.spines['right'].set_color('none')
        # ax.spines['bottom'].set_position('zero')
        # ax.spines['top'].set_color('none')
        # ax.spines['left'].set_smart_bounds(True)
        # ax.spines['bottom'].set_smart_bounds(True)
        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')   

        # ax.set_frame_on(False)
        # ax.axes.get_yaxis().set_visible(False)
        # ax.axes.get_xaxis().set_visible(False)
        # ax.text(-1.5,-0.14,'$E_c$', fontsize=42)
        # ax.text(1,-0.14,'$E\'_c$', fontsize=42)
        # ax.text(-5,0.9,'\\textbf{DOS}', fontsize=44)
        # plt.fill_between(engy2,0,dos2,color='blue',alpha=0.3)
        # plt.fill_between(engy3,0,dos3, color='red',alpha=0.3)
        # plt.tight_layout(pad=3)
        plt.tight_layout(pad=3)
        cwd=os.getcwd()
        # os.chdir('Presentation')
        plt.savefig('prob_dist.pdf')
        os.chdir(cwd)
# plot_loc_ext()


# plot_box()
plot_box()