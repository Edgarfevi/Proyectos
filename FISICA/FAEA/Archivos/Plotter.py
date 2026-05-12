from __future__ import print_function

import os
import warnings as wr

import ROOT as r

from Selector import Selector


class Plotter:
    """Gestiona la lectura de procesos, el apilado de histogramas y los conteos."""

    def __init__(self, backgrounds, data="", path="./results"):
        self.data = data
        self.backgrounds = backgrounds
        self.savepath = path

        self.listOfSelectors = []
        self.colors = []
        self.xtitle = ""
        self.ytitle = ""

        # Cargamos un selector por cada proceso de fondo.
        counter = 0
        for process in self.backgrounds:
            self.listOfSelectors.append(Selector(process))
            self.colors.append(counter + 1)
            counter += 1

        self.dataSelector = Selector(self.data) if self.data != "" else None

    def SetColors(self, col):
        self.colors = col

    def SetXtitle(self, tit):
        self.xtitle = tit

    def SetYtitle(self, tit):
        self.ytitle = tit

    def GetHisto(self, process, name):
        """Devuelve el histograma pedido para un proceso concreto."""

        if process == self.data and self.dataSelector:
            return self.dataSelector.GetHisto(name)

        for selector in self.listOfSelectors:
            if process == selector.name:
                return selector.GetHisto(name)

        wr.warn(
            "[Plotter::GetHisto] WARNING: histogram {} for process {} not found!".format(name, process)
        )
        return r.TH1F()

    def GetEvents(self, process, name):
        """Devuelve el número de eventos integrando el histograma correspondiente."""

        return self.GetHisto(process, name).Integral()

    def Stack(self, name):
        """Dibuja un histograma apilado o, en el caso de eficiencia, una curva tipo punto."""

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        # Caso especial: la eficiencia del trigger se representa como un gráfico independiente.
        if "Eficiencia" in name:
            canvas = r.TCanvas("c_" + name, "", 800, 600)

            # El guion pide usar la simulación de ttbar para esta curva.
            h_eff = self.GetHisto("ttbar", name)
            h_eff.SetMarkerStyle(20)
            h_eff.SetMarkerColor(r.kRed + 1)
            h_eff.SetLineColor(r.kRed + 1)
            h_eff.SetTitle("Eficiencia del Trigger (MC ttbar);p_{T}^{#mu} (GeV);Eficiencia")
            h_eff.SetMaximum(1.2)
            h_eff.SetMinimum(0.0)
            h_eff.Draw("PE")

            canvas.Print(self.savepath + "/" + name + ".png")
            canvas.Close()
            return

        canvas = r.TCanvas("c_" + name, "", 800, 600)
        hstack = r.THStack("hstack_" + name, "")
        legend = r.TLegend(0.75, 0.6, 0.89, 0.89)
        legend.SetBorderSize(0)

        for i, selector in enumerate(self.listOfSelectors):
            histo = selector.GetHisto(name)
            histo.SetFillColor(self.colors[i])
            histo.SetLineColor(1)
            hstack.Add(histo)
            legend.AddEntry(histo, selector.name, "f")

        hstack.SetMaximum(hstack.GetMaximum() * 1.2)
        hstack.Draw("hist")

        if self.xtitle != "":
            hstack.GetXaxis().SetTitle(self.xtitle)
        if self.ytitle != "":
            hstack.GetYaxis().SetTitle(self.ytitle)

        if self.dataSelector:
            hdata = self.dataSelector.GetHisto(name)
            hdata.SetMarkerStyle(20)
            hdata.Draw("pesame")
            legend.AddEntry(hdata, "data", "p")

        legend.Draw()
        canvas.Print(self.savepath + "/" + name + ".png")
        canvas.Close()

    def PrintCounts(self, name):
        """Imprime por pantalla el conteo de eventos para un histograma dado."""

        print("\nPrinting events for {}:".format(name))
        for selector in self.listOfSelectors:
            print("{}: {:.2f}".format(selector.name, selector.GetHisto(name).Integral()))
        if self.dataSelector:
            print("data: {:.2f}".format(self.dataSelector.GetHisto(name).Integral()))

    def SaveCounts(self, name, overridename=""):
        """Guarda en texto los yields del histograma indicado."""

        filename = overridename if overridename else "yields_" + name
        if ".txt" not in filename:
            filename += ".txt"

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        yields_mc = {}
        total_bkg, n_signal = 0.0, 0.0

        for selector in self.listOfSelectors:
            val = selector.GetHisto(name).Integral()
            yields_mc[selector.name] = val
            if selector.name == "ttbar":
                n_signal = val
            else:
                total_bkg += val

        n_obs = self.dataSelector.GetHisto(name).Integral() if self.dataSelector else 0.0

        # Se asume el orden definido en el script principal: ttbar ocupa el último lugar.
        if self.listOfSelectors[7].eficiencia_b is not None:
            self.eff_b = self.listOfSelectors[7].eficiencia_b

        with open(self.savepath + "/" + filename, "w") as f:
            f.write("====================================================\n")
            f.write("      INFORME DE EVENTOS: {:<20}\n".format(name))
            f.write("====================================================\n\n")
            f.write("{:<20} | {:>15}\n".format("PROCESO", "EVENTOS"))
            f.write("-" * 40 + "\n")

            for proc, val in yields_mc.items():
                if proc != "ttbar":
                    f.write("{:<20} | {:>15.2f}\n".format(proc, val))

            f.write("-" * 40 + "\n")
            f.write("{:<20} | {:>15.2f}\n".format("TOTAL FONDO (MC)", total_bkg))
            f.write("{:<20} | {:>15.2f}\n".format("SENAL EXPECT. (tt)", n_signal))
            f.write("-" * 40 + "\n")
            f.write("{:<20} | {:>15.2f}\n".format("OBSERVADO (DATA)", n_obs))
            f.write("====================================================\n\n")

            if total_bkg + n_signal > 0:
                f.write("Pureza esperada (S/S+B): {:.2f}%\n".format(n_signal / (total_bkg + n_signal) * 100))
            f.write("Exceso de datos (S_neta): {:.2f} eventos\n".format(n_obs - total_bkg))