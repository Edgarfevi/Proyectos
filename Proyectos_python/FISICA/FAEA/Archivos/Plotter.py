from __future__ import print_function
import warnings as wr
import os
from Selector import Selector
import ROOT as r

class Plotter:
    ''' Class to draw histograms and get info from Selector'''
    ### =============================================
    ### Constructor
    def __init__(self, backgrounds, data = '', path = "/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/FAEA/Resultados"):
        ''' Initialize a new plotter... give a list with all names of MC samples and the name of the data sample '''
        self.data = data
        self.backgrounds = backgrounds
        self.savepath = path
        counter = 0
        for p in self.backgrounds:
            self.listOfSelectors.append(Selector(p))
            self.colors.append(counter+1)
            counter += 1

        if (self.data != ''): self.dataSelector = Selector(self.data)
        return

    ### =============================================
    ### Attributes
    savepath = "."
    listOfSelectors = []
    dataSelector = Selector()
    data = ''
    colors = []

    # Default parameters
    fLegX1, fLegY1, fLegX2, fLegY2 = 0.75, 0.55, 0.89, 0.89
    LegendTextSize  = 0.035
    xtitle = ''
    ytitle = ''
    title  = ''


    ### =============================================
    ### Methods
    def SetLegendPos(self, x1, y1, x2, y2):
        ''' Change the default position of the legend'''
        self.fLegX1 = x1
        self.fLegY1 = y1
        self.fLegX2 = x2
        self.fLegY2 = y2


    def SetLegendSize(self, t = 0.065):
        ''' Change the default size of the text in the legend'''
        self.LegendTextSize = t


    def SetSavePath(self, newpath):
        '''Change where plots and text dumps are going to be saved. By default: ./results'''
        self.savepath = newpath


    def SetColors(self, col):
        ''' Set the colors for each MC sample '''
        self.colors = col


    def SetTitle(self, tit):
        ''' Set title of the plot '''
        self.title = tit


    def SetXtitle(self, tit):
        ''' Set title of X axis '''
        self.xtitle = tit


    def SetYtitle(self, tit):
        ''' Set title of Y axis '''
        self.ytitle = tit


    def GetHisto(self, process, name):
        ''' Returns histogram 'name' for a given process '''
        for s in self.listOfSelectors:
            if name not in s.name: continue
            h = s.GetHisto(name)
            return h

        wr.warn("[Plotter::GetHisto] WARNING: histogram {h} for process {p} not found!".format(h = name, p = process))
        return r.TH1F()


    def GetEvents(self, process, name):
        ''' Returns the integral of a histogram '''
        return self.GetHisto(process, name).Integral()


    def Stack(self, name):
        ''' Produce a stack plot for a variable given '''
        if (isinstance(name, list)):
            for nam in name: self.Stack(nam)
            return

        c = r.TCanvas('c_'+name, 'c', 10, 10, 800, 600)
        l = r.TLegend(self.fLegX1, self.fLegY1, self.fLegX2, self.fLegY2)
        l.SetTextSize(self.LegendTextSize)
        l.SetBorderSize(0)
        l.SetFillColor(10)

        hstack = r.THStack('hstack_' + name, "hstack")
        counter = 0
        for s in self.listOfSelectors:
            h = s.GetHisto(name)
            h.SetFillColor(self.colors[counter])
            h.SetLineColor(0)
            hstack.Add(h)
            l.AddEntry(h, s.name, "f")
            counter += 1
        hstack.Draw("hist")

        if self.title  != '': hstack.SetTitle(self.title)
        if self.xtitle != '': hstack.GetXaxis().SetTitle(self.xtitle)
        if self.ytitle != '': hstack.GetYaxis().SetTitle(self.ytitle)
        hstack.GetYaxis().SetTitleOffset(1.35)
        Max = hstack.GetStack().Last().GetMaximum();

        if self.data != '':
            hdata = self.dataSelector.GetHisto(name)
            hdata.SetMarkerStyle(20)
            hdata.SetMarkerColor(1)
            hdata.Draw("pesame")
            MaxData = hdata.GetMaximum()
            if(Max < MaxData): Max = MaxData
            l.AddEntry(hdata, self.dataSelector.name, 'p')
        l.Draw("same")
        hstack.SetMaximum(Max * 1.1)
        create_folder(self.savepath)
        c.Print(self.savepath + "/" + name + '.png', 'png')
        c.Print(self.savepath + "/" + name + '.pdf', 'pdf')
        c.Close()
        return


    def PrintCounts(self, name):
        ''' Print the number of events for each sample in a given histogram '''
        if (isinstance(name, list)):
            for nam in name: self.PrintEvents(nam)
            return

        print("\nPrinting number of events for histogram {h}:".format(h = name))
        print('----------------------------------------------------')
        total = 0.
        for s in self.listOfSelectors:
            h = s.GetHisto(name)
            print("{nam}: {num}".format(nam = s.name, num = h.Integral()))
            total += h.Integral()

        print('Expected (MC): {tot}'.format(tot = total))
        print('------------------------------')
        if self.data != '':
            hdata = self.dataSelector.GetHisto(name)
            print('Observed (data): {tot}'.format(tot = hdata.Integral()))
            print('------------------------------')
        return


    def SaveCounts(self, name, overridename = ""):
        ''' Save in a text file the number of events for each sample in a given histogram '''
        if (isinstance(name, list)):
            for nam in name: self.SaveCounts(nam, overridename = overridename)
            return

        filename = "yields_{h}".format(h = name) if (overridename == "") else overridename
        create_folder(self.savepath)
        outfile = open(self.savepath + "/" + filename + (".txt" if ".txt" not in overridename else ""), "w")

        thelines = []

        thelines.append("Number of events for histogram {h}:\n".format(h = name))
        thelines.append("----------------------------------------------------\n")
        total = 0.
        for s in self.listOfSelectors:
            h = s.GetHisto(name)
            thelines.append("{nam}: {num}\n".format(nam = s.name, num = h.Integral()))
            total += h.Integral()
        thelines.append('Expected (MC): {tot}\n'.format(tot = total))
        thelines.append('------------------------------\n')
        if self.data != '':
            hdata = self.dataSelector.GetHisto(name)
            thelines.append('Observed (data): {tot}\n'.format(tot = hdata.Integral()))
            thelines.append('------------------------------\n')

        outfile.writelines(thelines)
        outfile.close()
        return


def create_folder(path):
    if not os.path.exists(path): os.system("mkdir -p " + path)
    return
