import ROOT as rt
from pathlib import Path
import pandas as pd
import os

class total_comparation:
    """Construye figuras comparativas entre gen, reco y data para ambos canales.

    A partir de los archivos de eventos generados por la clase `construccion_individually`
    para cada tipo (gen, reco, data) y canal (ee, mumu), esta clase:
    1. Lee los archivos CSV con las variables cinemáticas y masa invariante.
    2. Genera figuras comparativas superpuestas de pT y eta para cada partícula.
    3. Crea histogramas comparativos de la masa invariante del sistema dileptónico.
    4. Guarda todas las figuras en formato PNG en el directorio `img/`.
    """

    def __init__(self):
        """Inicializa la construcción de figuras comparativas."""
        self.BASE_DIR = Path(__file__).resolve().parent
        self.load_all_dataframes()


    def load_all_dataframes(self):
        """Carga todos los archivos CSV de eventos desde el directorio results/.
        
        Lee archivos para cada combinación de tipo (gen/reco/data) y canal (ee/mumu).
        Los DataFrames se almacenan como atributos dinámicos en self para acceso rápido.
        
        Atributos creados:
            - {type}_{p1_name}_{p2_name}_df: DataFrame con datos de eventos filtrados
        """
        # Iteramos sobre cada tipo de muestra y cada canal de leptones
        for type in ["gen", "reco", "data"]:
            for p1_name, p2_name in [("electron", "positron"), ("muon", "anti-muon")]:
                # Construimos la ruta esperada del archivo CSV
                csv_path = self.BASE_DIR / "results" / f"events_data_{type}_{p1_name}_{p2_name}.csv"
                
                # Si el archivo existe, lo cargamos en memoria
                if csv_path.exists():
                    df = pd.read_csv(csv_path, index_col="event_index")
                    setattr(self, f"{type}_{p1_name}_{p2_name}_df", df)
                else:
                    print(f"Advertencia: No se encontró el archivo {csv_path}.")

    def _quantile_bin_range(self, hist, central_fraction=0.995):
        """Devuelve el rango de bins que contiene la fracción central de eventos.
        
        Implementa un algoritmo de cuantiles que descarta las colas de la distribución
        simétricamente, manteniendo únicamente los datos centrales especificados.
        
        Parámetros:
            hist (ROOT.TH1F): Histograma del cual extraer el rango.
            central_fraction (float): Fracción de eventos a retener (default: 0.995 = 99.5%).
        
        Retorna:
            tuple: (low_bin, high_bin) con los índices de bins que delimitan la región central.
                   None si el histograma está vacío.
        """
        # Total de bins en el histograma
        n_bins = hist.GetNbinsX()
        # Integral total de eventos
        total = hist.Integral(1, n_bins)
        
        # Si no hay eventos, retornar None
        if total <= 0.0:
            return None

        # Calcular la fracción de eventos que descartamos en cada cola
        tail_fraction = max(0.0, min(0.5, (1.0 - central_fraction) / 2.0))
        # Número de eventos a descartar en cada cola
        tail_events = total * tail_fraction

        # Buscar el bin inferior que supera el umbral de eventos de cola
        low_bin = 1
        cumulative = 0.0
        while low_bin <= n_bins:
            next_cumulative = cumulative + hist.GetBinContent(low_bin)
            if next_cumulative > tail_events:
                break
            cumulative = next_cumulative
            low_bin += 1

        # Buscar el bin superior que supera el umbral de eventos de cola (desde atrás)
        high_bin = n_bins
        cumulative = 0.0
        while high_bin >= 1:
            next_cumulative = cumulative + hist.GetBinContent(high_bin)
            if next_cumulative > tail_events:
                break
            cumulative = next_cumulative
            high_bin -= 1

        # Validar y ajustar rangos
        low_bin = max(1, min(low_bin, n_bins))
        high_bin = max(1, min(high_bin, n_bins))
        if low_bin > high_bin:
            low_bin, high_bin = 1, n_bins

        return low_bin, high_bin

    def _autofit_mass_pair(self, hist1, hist2, central_fraction=0.995, y_padding=1.25):
        """Ajusta ejes para histogramas de masa invariante usando fracción central.
        
        Aplica el rango cuantílico a ambos histogramas simultáneamente para garantizar
        que el eje X común capture la región de interés física. El eje Y se escala
        con padding para mejorar la visualización.
        
        Parámetros:
            hist1, hist2 (ROOT.TH1F): Histogramas a ajustar.
            central_fraction (float): Fracción de eventos centrales a mostrar (default: 0.995).
            y_padding (float): Factor de amplitud del eje Y (default: 1.25).
        """
        # Obtener rangos cuantílicos de ambos histogramas
        range1 = self._quantile_bin_range(hist1, central_fraction=central_fraction)
        range2 = self._quantile_bin_range(hist2, central_fraction=central_fraction)

        x_mins = []
        x_maxs = []

        # Recopilar los límites X de ambos histogramas
        if range1 is not None:
            b1_min, b1_max = range1
            x_mins.append(hist1.GetXaxis().GetBinLowEdge(b1_min))
            x_maxs.append(hist1.GetXaxis().GetBinUpEdge(b1_max))

        if range2 is not None:
            b2_min, b2_max = range2
            x_mins.append(hist2.GetXaxis().GetBinLowEdge(b2_min))
            x_maxs.append(hist2.GetXaxis().GetBinUpEdge(b2_max))

        # Aplicar rango X común (unión de ambos rangos)
        if x_mins and x_maxs:
            x_min = min(x_mins)
            x_max = max(x_maxs)
            hist1.GetXaxis().SetRangeUser(x_min, x_max)
            hist2.GetXaxis().SetRangeUser(x_min, x_max)

        # Escalar eje Y con padding para visualización clara
        y_max = max(hist1.GetMaximum(), hist2.GetMaximum())
        hist1.SetMinimum(0.0)
        hist2.SetMinimum(0.0)
        hist1.SetMaximum(max(1e-6, y_max * y_padding))

    def _autofit_overlay_pair(self, hist1, hist2, central_fraction=0.995, y_padding=1.25):
        """Ajusta ejes para overlays usando la fracción central de datos.
        
        Similar a _autofit_mass_pair pero usado específicamente para superponer
        histogramas en la misma visualización. Asegura que ambos histogramas
        compartan el mismo rango X basado en sus cuantiles centrales.
        
        Parámetros:
            hist1, hist2 (ROOT.TH1F): Histogramas a superponer.
            central_fraction (float): Fracción de eventos centrales (default: 0.995).
            y_padding (float): Factor de escala del eje Y (default: 1.25).
        """
        # Calcular rangos cuantílicos independientes para cada histograma
        range1 = self._quantile_bin_range(hist1, central_fraction=central_fraction)
        range2 = self._quantile_bin_range(hist2, central_fraction=central_fraction)

        x_mins = []
        x_maxs = []

        # Recopilar límites X de ambas distribuciones
        if range1 is not None:
            b1_min, b1_max = range1
            x_mins.append(hist1.GetXaxis().GetBinLowEdge(b1_min))
            x_maxs.append(hist1.GetXaxis().GetBinUpEdge(b1_max))

        if range2 is not None:
            b2_min, b2_max = range2
            x_mins.append(hist2.GetXaxis().GetBinLowEdge(b2_min))
            x_maxs.append(hist2.GetXaxis().GetBinUpEdge(b2_max))

        # Usar rango X que abarca ambos histogramas para comparación justa
        if x_mins and x_maxs:
            x_min = min(x_mins)
            x_max = max(x_maxs)
            hist1.GetXaxis().SetRangeUser(x_min, x_max)
            hist2.GetXaxis().SetRangeUser(x_min, x_max)

        # Normalizar eje Y a partir del máximo de ambos histogramas
        y_max = max(hist1.GetMaximum(), hist2.GetMaximum())
        hist1.SetMinimum(0.0)
        hist2.SetMinimum(0.0)
        hist1.SetMaximum(max(1e-6, y_max * y_padding))

    def plot_invariant_mass_overlay_channels(self, normalize=False):
        """Superpone canales ee y mumu para cada tipo (gen, reco, data).
        
        Genera un canvas dividido en 1x3 con un pad por tipo de muestra:
            - Pad 1: Gen (eventos de generador Monte Carlo - verdad)
            - Pad 2: Reco (eventos reconstruidos por el detector)
            - Pad 3: Data (eventos de datos experimentales reales)
        
        En cada pad se dibujan superpuestos los histogramas de masa invariante
        para los canales ee (electrón-positrón) y μμ (muón-antimuón).
        
        Parámetros:
            normalize (bool): Si True, normaliza los histogramas por integral.
                             Útil para comparar formas independientemente de yields.
        
        Salida:
            PNG: invariant_mass_overlay_channels_by_type[_norm].png
        """
        self.free_memory()  # Liberamos memoria antes de crear nuevos objetos gráficos
        
        os.makedirs(self.BASE_DIR / "img", exist_ok=True)

        # Estilos visuales para cada canal
        channel_styles = {
            ("electron", "positron"): {"color": rt.kAzure + 1, "label": "Canal ee"},
            ("muon", "anti-muon"): {"color": rt.kOrange + 7, "label": "Canal mumu"},
        }

        channels = [("electron", "positron"), ("muon", "anti-muon")]
        sample_types = ["gen", "reco", "data"]

        # Crear canvas con 3 pads verticales
        canvas = rt.TCanvas(
            "canvas_invariant_mass_overlay_channels",
            "Masa invariante Z: canales ee y mumu por tipo",
            1200,
            1600,
        )
        canvas.Divide(1, 3)

        all_hists = []
        all_legends = []

        # Iterar sobre cada tipo de muestra (gen/reco/data)
        for pad_idx, type in enumerate(sample_types, start=1):
            canvas.cd(pad_idx)
            rt.gPad.SetGridx()
            rt.gPad.SetGridy()

            type_hists = []
            y_max = 0.0

            # Para cada canal, crear histograma de masa invariante
            for p1_name, p2_name in channels:
                df = getattr(self, f"{type}_{p1_name}_{p2_name}_df", None)
                if df is None or "invariant_mass_Z" not in df.columns:
                    print(f"Datos no disponibles para {type} {p1_name}-{p2_name}.")
                    continue

                # Crear histograma y llenar con datos
                hist = rt.TH1F(
                    f"hist_mz_overlay_{type}_{p1_name}_{p2_name}",
                    "",
                    100,
                    0,
                    200,
                )
                for value in df["invariant_mass_Z"].dropna().values:
                    hist.Fill(float(value))

                # Normalizar si está solicitado
                if normalize and hist.Integral() > 0.0:
                    hist.Scale(1.0 / hist.Integral())

                # Aplicar estilos
                hist.SetLineColor(channel_styles[(p1_name, p2_name)]["color"])
                hist.SetLineWidth(2)
                hist.SetFillStyle(0)
                hist.SetStats(0)

                y_max = max(y_max, hist.GetMaximum())
                type_hists.append((p1_name, p2_name, hist))
                all_hists.append(hist)

            # Manejo de casos sin datos
            if not type_hists:
                text = rt.TLatex()
                text.SetTextSize(0.04)
                text.DrawLatexNDC(0.20, 0.50, f"Sin datos para {type}")
                all_hists.append(text)
                continue

            # Ajustar título según normalización
            y_title = "Eventos normalizados" if normalize else "Eventos"
            title = f"Masa invariante Z ({type});m_{{ll}} (GeV/c^{{2}});{y_title}"

            # Dibujar primer histograma (establece el frame)
            _, _, first_hist = type_hists[0]
            first_hist.SetTitle(title)
            
            # Ajustar ejes automáticamente usando cuantiles
            if len(type_hists) > 1:
                _, _, second_hist = type_hists[1]
                self._autofit_mass_pair(first_hist, second_hist, central_fraction=0.995, y_padding=1.25)
            else:
                first_hist.SetMaximum(max(1e-6, y_max * 1.25))
            
            first_hist.Draw("HIST")

            # Dibujar histogramas adicionales superpuestos
            for _, _, hist in type_hists[1:]:
                hist.Draw("HIST SAME")

            # Crear leyenda
            legend = rt.TLegend(0.68, 0.70, 0.90, 0.88)
            legend.SetBorderSize(0)
            for p1_name, p2_name, hist in type_hists:
                legend.AddEntry(hist, channel_styles[(p1_name, p2_name)]["label"], "l")
            legend.Draw()
            all_legends.append(legend)

        # Guardar canvas
        canvas.Draw()
        output_name = "invariant_mass_overlay_channels_by_type_norm.png" if normalize else "invariant_mass_overlay_channels_by_type.png"
        canvas.SaveAs(str(self.BASE_DIR / "img" / output_name))

        # Almacenar referencias para evitar garbage collection
        self.canvas_invariant_mass_overlay_channels = canvas
        self.legend_invariant_mass_overlay_channels = all_legends
        self.hists_invariant_mass_overlay_channels = all_hists

        if hasattr(self, "_keep_alive"):
            self._keep_alive.update(locals())
        else:
            self._keep_alive = locals().copy()

        print(f"Grafica generada: {output_name}")

    def plot_invariant_mass_overlay_types_by_channel(self, normalize=False):
        """Canvas 2x1 por canal superponiendo gen, reco y data en cada cuadro.
        
        Genera un canvas con dos pads lado a lado:
            - Pad 1 (izquierda): Canal ee (electron-positron)
            - Pad 2 (derecha): Canal mumu (muon-anti-muon)
        
        En cada pad se superponen los histogramas de masa invariante para
        gen (verdad), reco (reconstructo) y data (experimental).
        Esto permite comparar el efecto de la reconstrucción y validar contra datos reales.
        
        Parámetros:
            normalize (bool): Si True, normaliza histogramas por integral.
        
        Salida:
            PNG: invariant_mass_overlay_types_by_channel[_norm].png
        """
        self.free_memory()  # Liberamos memoria antes de crear nuevos objetos gráficos
        
        os.makedirs(self.BASE_DIR / "img", exist_ok=True)

        # Estilos visuales para cada tipo de muestra
        type_styles = {
            "gen": {"color": rt.kAzure + 1, "label": "Gen"},
            "reco": {"color": rt.kOrange + 7, "label": "Reco"},
            "data": {"color": rt.kGreen + 2, "label": "Data"},
        }

        channels = [
            ("electron", "positron", "Canal ee"),
            ("muon", "anti-muon", "Canal mumu"),
        ]

        # Crear canvas con 2 pads horizontales
        canvas = rt.TCanvas(
            "canvas_invariant_mass_overlay_types_by_channel",
            "Masa invariante Z: gen/reco/data por canal",
            1400,
            1200,
        )
        canvas.Divide(2, 1)

        all_hists = []
        all_legends = []

        # Iterar sobre cada canal (ee y mumu)
        for pad_idx, (p1_name, p2_name, channel_label) in enumerate(channels, start=1):
            canvas.cd(pad_idx)
            rt.gPad.SetGridx()
            rt.gPad.SetGridy()

            channel_hists = []
            y_max = 0.0

            # Para cada tipo de muestra, crear histograma
            for type in ["gen", "reco", "data"]:
                df = getattr(self, f"{type}_{p1_name}_{p2_name}_df", None)
                if df is None or "invariant_mass_Z" not in df.columns:
                    print(f"Datos no disponibles para {type} {p1_name}-{p2_name}.")
                    continue

                # Crear y llenar histograma
                hist = rt.TH1F(
                    f"hist_mz_overlay_types_{type}_{p1_name}_{p2_name}",
                    "",
                    100,
                    0,
                    200,
                )
                for value in df["invariant_mass_Z"].dropna().values:
                    hist.Fill(float(value))

                # Normalizar si lo indica el parámetro
                if normalize and hist.Integral() > 0.0:
                    hist.Scale(1.0 / hist.Integral())

                # Aplicar estilos
                hist.SetLineColor(type_styles[type]["color"])
                hist.SetLineWidth(2)
                hist.SetFillStyle(0)
                hist.SetStats(0)

                y_max = max(y_max, hist.GetMaximum())
                channel_hists.append((type, hist))
                all_hists.append(hist)

            # Manejo de casos sin datos
            if not channel_hists:
                text = rt.TLatex()
                text.SetTextSize(0.04)
                text.DrawLatexNDC(0.18, 0.50, f"Sin datos para {channel_label}")
                all_hists.append(text)
                continue

            # Ajustar título según normalización
            y_title = "Eventos normalizados" if normalize else "Eventos"
            title = f"{channel_label};m_{{ll}} (GeV/c^{{2}});{y_title}"

            # Dibujar primer histograma
            _, first_hist = channel_hists[0]
            first_hist.SetTitle(title)
            
            # Ajustar ejes usando cuantiles si hay múltiples histogramas
            if len(channel_hists) > 1:
                _, second_hist = channel_hists[1]
                self._autofit_mass_pair(first_hist, second_hist, central_fraction=0.995, y_padding=1.25)
            else:
                first_hist.SetMaximum(max(1e-6, y_max * 1.25))
            
            first_hist.Draw("HIST")

            # Superponer histogramas adicionales
            for _, hist in channel_hists[1:]:
                hist.Draw("HIST SAME")

            # Crear leyenda
            legend = rt.TLegend(0.68, 0.70, 0.90, 0.88)
            legend.SetBorderSize(0)
            for type, hist in channel_hists:
                legend.AddEntry(hist, type_styles[type]["label"], "l")
            legend.Draw()
            all_legends.append(legend)

        # Guardar canvas
        canvas.Draw()
        output_name = (
            "invariant_mass_overlay_types_by_channel_norm.png"
            if normalize
            else "invariant_mass_overlay_types_by_channel.png"
        )
        canvas.SaveAs(str(self.BASE_DIR / "img" / output_name))

        # Almacenar referencias
        self.canvas_invariant_mass_overlay_types_by_channel = canvas
        self.legend_invariant_mass_overlay_types_by_channel = all_legends
        self.hists_invariant_mass_overlay_types_by_channel = all_hists

        if hasattr(self, "_keep_alive"):
            self._keep_alive.update(locals())
        else:
            self._keep_alive = locals().copy()

        print(f"Grafica generada: {output_name}")
    
    def plot_comparative_figures_individually(self):
        """Genera figuras comparativas individuales por partícula (electron/positron/muon/anti-muon).
        
        Para cada partícula en cada canal (ee y mumu), crea dos canvases:
        
        1. CANVAS DE RATIOS (comparative_ratios_{particle}.png):
           - Compara tres ratios: Reco/Gen, Data/Reco, Data/Gen
           - Cada ratio muestra dos variables: pT (momento transverso) y η (pseudorapidez)
           - Layout: 3 filas (ratios) × 2 columnas (variables) = 6 pads
           - Incluye línea de referencia roja en y=1
           - Útil para detectar desviaciones sistemáticas en reconstrucción
        
        2. CANVAS DE OVERLAYS (comparative_overlay_{particle}.png):
           - Superpone histogramas normalizados para comparar formas
           - Mismo layout: 3 filas × 2 columnas
           - Ayuda a identificar diferencias en resolución y eficiencia
        
        Los histogramas se normalizan antes de calcular ratios para comparar
        distribuciones de forma, no yields absolutos.
        
        Salida:
            PNG: comparative_ratios_{particle}.png
            PNG: comparative_overlay_{particle}.png
        """
        self.free_memory()  # Liberamos memoria antes de crear nuevos objetos gráficos
        
        def _bin_range_with_content(hist, include_errors=False):
            """Devuelve el rango de bins con contenido útil.
            
            Escanea el histograma para encontrar el primer y último bin
            que contiene contenido significativo (útil para ratios ruidosos).
            """
            first_bin = None
            last_bin = None
            for bin_idx in range(1, hist.GetNbinsX() + 1):
                content = hist.GetBinContent(bin_idx)
                error = hist.GetBinError(bin_idx)
                if content != 0.0 or (include_errors and error != 0.0):
                    if first_bin is None:
                        first_bin = bin_idx
                    last_bin = bin_idx
            if first_bin is None:
                return None
            return first_bin, last_bin

        def _autofit_ratio_hist(hist, y_padding=1.20):
            """Ajusta ejes X/Y del ratio al rango con contenido.
            
            Para histogramas de ratios, centra el eje Y alrededor de y=1
            (referencia de agreement) y lo expande simétricamente.
            """
            range_info = _bin_range_with_content(hist, include_errors=True)
            if range_info is None:
                hist.GetYaxis().SetRangeUser(0.5, 1.5)
                return

            first_bin, last_bin = range_info
            x_min = hist.GetXaxis().GetBinLowEdge(first_bin)
            x_max = hist.GetXaxis().GetBinUpEdge(last_bin)
            hist.GetXaxis().SetRangeUser(x_min, x_max)

            y_min = float("inf")
            y_max = float("-inf")
            for bin_idx in range(first_bin, last_bin + 1):
                content = hist.GetBinContent(bin_idx)
                error = hist.GetBinError(bin_idx)
                if content == 0.0 and error == 0.0:
                    continue
                y_min = min(y_min, content - error)
                y_max = max(y_max, content + error)

            if y_min == float("inf") or y_max == float("-inf"):
                y_min, y_max = 0.5, 1.5

            y_min = min(y_min, 1.0)
            y_max = max(y_max, 1.0)
            span = max(1e-6, y_max - y_min)
            margin = span * (y_padding - 1.0)
            hist.GetYaxis().SetRangeUser(y_min - margin, y_max + margin)

        # =========================================================================
        # PARTE 1: INICIALIZAR HISTOGRAMAS PARA TODAS LAS COMBINACIONES
        # =========================================================================
        # Creamos histogramas individuales para cada partícula, tipo de muestra,
        # y variable (pT, eta). Esto prepara los datos para ratios y overlays.
        
        for plot_type in ["pt", "eta"]:
            for p1_name, p2_name in [("electron", "positron"), ("muon", "anti-muon")]:
                for type in ["gen", "reco", "data"]:
                    df = getattr(self, f"{type}_{p1_name}_{p2_name}_df", None)
                    if df is not None and plot_type == "pt":
                        hist = rt.TH1F(
                            f"hist_invariant_mass_{type}_{p1_name}_{p2_name}_{plot_type}",
                            f" ({type});m_{{ll}} (GeV/c^{{2}});Eventos",
                            150,
                            0,
                            500,
                        )
                        hist
                        for value in df[f"{plot_type}_{p1_name}"].dropna().values:
                            hist.Fill(value)
                        hist2 = rt.TH1F(
                            f"hist_invariant_mass_{type}_{p1_name}_{p2_name}_{plot_type}_p2",
                            f" ({type});m_{{ll}} (GeV/c^{{2}});Eventos",
                            150,
                            0,
                            500,
                        )
                        for value in df[f"{plot_type}_{p2_name}"].dropna().values:
                            hist2.Fill(value)
                        setattr(self, f"hist_{type}_{p1_name}_{plot_type}", hist)
                        setattr(self, f"hist_{type}_{p2_name}_{plot_type}", hist2)
                        hist_canal = rt.TH1F(f"pt_canal_{type}_{p1_name}_{p2_name}", f" ({type});m_{{ll}} (GeV/c^{{2}});Eventos", 150, 0, 500)
                    elif df is not None and plot_type == "eta":
                        hist = rt.TH1F(
                            f"{type}_{p1_name}_{plot_type}",
                            f" ({type});m_{{ll}} (GeV/c^{{2}});Eventos",
                            100,
                            -20,
                            20,
                        )
                        hist2 = rt.TH1F(
                            f"{type}_{p2_name}_{plot_type}",
                            f" ({type});m_{{ll}} (GeV/c^{{2}});Eventos",
                            100,
                            -20,
                            20,
                        )
                        hist_canal = rt.TH1F(f"eta_canal_{type}_{p1_name}_{p2_name}", f" ({type});m_{{ll}} (GeV/c^{{2}});Eventos", 100, -20, 20)
                        for value in df[f"{plot_type}_{p1_name}"].dropna().values:
                            hist.Fill(value)
                        for value in df[f"{plot_type}_{p2_name}"].dropna().values:
                            hist2.Fill(value)
                        for value in df[f"{plot_type}_canal"].dropna().values:
                            hist_canal.Fill(value)
                        setattr(self, f"hist_{type}_{p1_name}_{plot_type}", hist)
                        setattr(self, f"hist_{type}_{p2_name}_{plot_type}", hist2)
                        setattr(self, f"hist_{type}_{p1_name}_{p2_name}_{plot_type}", hist_canal)
                    else:
                        print(f"Datos no disponibles para {type} {p1_name} {p2_name}.")
        
        # =========================================================================
        # PARTE 2: CREAR CANVASES DE RATIOS Y OVERLAYS POR PARTÍCULA
        # =========================================================================
        # Para cada partícula individual, generamos dos canvases:
        # 1. Canvas de ratios (para detectar diferencias relativas)
        # 2. Canvas de overlays (para comparar formas de distribuciones)
        
        canvas_ratios = rt.TCanvas("canvas_comparative", "Comparacion de gen, reco y data", 1600, 1200)
        canvas_ratios.Divide(2, 3)
        
        # Iteramos por cada canal (ee y mumu)
        for p1_name, p2_name in [("electron", "positron"), ("muon", "anti-muon")]:
            
            # Para cada partícula en el canal, crear canvases individuales
            for current_particle in [p1_name, p2_name]:
                
                # Verificar que existan histogramas para esta partícula
                if not hasattr(self, f"hist_gen_{current_particle}_pt"):
                    continue 
                    
                # CANVAS 1: RATIOS (Reco/Gen, Data/Reco, Data/Gen)
                canvas_ratios = rt.TCanvas(
                    f"canvas_ratios_{current_particle}", 
                    f"Comparacion ratios {current_particle}", 
                    1600, 1200
                )
                canvas_ratios.Divide(2, 3)  # 2 columnas (pT, η) × 3 filas (ratios)

                # CANVAS 2: OVERLAYS (histogramas normalizados superpuestos)
                canvas_overlay = rt.TCanvas(
                    f"canvas_overlay_{current_particle}",
                    f"Comparacion overlay {current_particle}",
                    1600,
                    1200,
                )
                canvas_overlay.Divide(2, 3)
                
                # Definir los tres ratios a calcular
                ratios_to_plot = [
                    ("reco", "gen", "Reco / Gen"),
                    ("data", "reco", "Data / Reco"),
                    ("data", "gen", "Data / Gen")
                ]
                
                pad_counter = 1
                for num_type, den_type, ratio_title in ratios_to_plot:
                    for plot_type in ["pt", "eta"]:
                        canvas_ratios.cd(pad_counter)
                        
                        # Recuperar histogramas para numerador y denominador
                        h_num = getattr(self, f"hist_{num_type}_{current_particle}_{plot_type}", None)
                        h_den = getattr(self, f"hist_{den_type}_{current_particle}_{plot_type}", None)
                        
                        if h_num and h_den and h_den.Integral() > 0:
                            # Clonar histogramas originales
                            h_num_norm = h_num.Clone(f"norm_{num_type}_{current_particle}_{plot_type}")
                            h_den_norm = h_den.Clone(f"norm_{den_type}_{current_particle}_{plot_type}")

                            # Normalizar ambos histogramas por su integral
                            # Esto garantiza que el ratio compare FORMAS, no yields absolutos
                            if h_num_norm.Integral() > 0:
                                h_num_norm.Scale(1.0 / h_num_norm.Integral())
                            if h_den_norm.Integral() > 0:
                                h_den_norm.Scale(1.0 / h_den_norm.Integral())

                            # Calcular el ratio de histogramas normalizados
                            h_ratio = h_num_norm.Clone(f"ratio_{num_type}_{den_type}_{current_particle}_{plot_type}")
                            h_ratio.Divide(h_den_norm)
                            
                            # Estilizado del histograma de ratio
                            variable_title = "p_{{T}} (GeV/c)" if plot_type == "pt" else "#eta"
                            h_ratio.SetTitle(f"{ratio_title} de {plot_type} ({current_particle});{variable_title};{ratio_title}")
                            h_ratio.SetLineColor(rt.kBlack)
                            h_ratio.SetMarkerStyle(20)
                            h_ratio.SetMarkerSize(0.8)
                            h_ratio.SetStats(0)

                            # Ajustar automáticamente ejes del ratio
                            _autofit_ratio_hist(h_ratio)
                            
                            # Dibujar con errores (puntos con barras)
                            h_ratio.Draw("E")
                            
                            # Agregar línea de referencia en y=1 (acuerdo perfecto)
                            x_first = h_ratio.GetXaxis().GetFirst()
                            x_last = h_ratio.GetXaxis().GetLast()
                            x_min = h_ratio.GetXaxis().GetBinLowEdge(x_first)
                            x_max = h_ratio.GetXaxis().GetBinUpEdge(x_last)
                            line_ref = rt.TLine(x_min, 1, x_max, 1)
                            line_ref.SetLineColor(rt.kRed)
                            line_ref.SetLineStyle(2)
                            line_ref.Draw("same")
                            
                            # Almacenar referencias para evitar garbage collection
                            setattr(self, f"ratio_obj_{num_type}_{den_type}_{current_particle}_{plot_type}", h_ratio)
                            setattr(self, f"ratio_line_{num_type}_{den_type}_{current_particle}_{plot_type}", line_ref)
                        else:
                            print(f"Faltan datos o division por cero para {num_type}/{den_type} de {plot_type} ({current_particle})")
                            
                        pad_counter += 1

                # =================================================================
                # SEGUNDA ETAPA: CANVAS DE OVERLAYS (Formas normalizadas)
                # =================================================================
                
                pad_counter = 1
                for num_type, den_type, ratio_title in ratios_to_plot:
                    for plot_type in ["pt", "eta"]:
                        canvas_overlay.cd(pad_counter)

                        h_num = getattr(self, f"hist_{num_type}_{current_particle}_{plot_type}", None)
                        h_den = getattr(self, f"hist_{den_type}_{current_particle}_{plot_type}", None)

                        if h_num and h_den and h_num.Integral() > 0 and h_den.Integral() > 0:
                            # Clonar para overlay
                            h_num_overlay = h_num.Clone(f"overlay_{num_type}_{current_particle}_{plot_type}")
                            h_den_overlay = h_den.Clone(f"overlay_{den_type}_{current_particle}_{plot_type}")

                            # Normalizar para comparar formas
                            h_num_overlay.Scale(1.0 / h_num_overlay.Integral())
                            h_den_overlay.Scale(1.0 / h_den_overlay.Integral())

                            variable_title = "p_{T} (GeV/c)" if plot_type == "pt" else "#eta"
                            h_num_overlay.SetTitle(
                                f"{ratio_title} de {plot_type} ({current_particle});{variable_title};Eventos normalizados"
                            )
                            h_num_overlay.SetLineColor(rt.kAzure + 1)
                            h_num_overlay.SetLineWidth(2)
                            h_num_overlay.SetStats(0)
                            h_num_overlay.SetFillStyle(0)

                            h_den_overlay.SetLineColor(rt.kOrange + 7)
                            h_den_overlay.SetLineWidth(2)
                            h_den_overlay.SetStats(0)
                            h_den_overlay.SetFillStyle(0)

                            # Aplicar ajuste de ejes usando cuantiles centrales (99.5%)
                            self._autofit_overlay_pair(h_num_overlay, h_den_overlay, central_fraction=0.995, y_padding=1.25)
                            
                            # Dibujar con histogramas (líneas continuas)
                            h_num_overlay.Draw("HIST")
                            h_den_overlay.Draw("HIST SAME")

                            # Crear leyenda para identificar histogramas
                            legend_overlay = rt.TLegend(0.68, 0.70, 0.90, 0.88)
                            legend_overlay.SetBorderSize(0)
                            legend_overlay.AddEntry(h_num_overlay, f"{num_type}", "l")
                            legend_overlay.AddEntry(h_den_overlay, f"{den_type}", "l")
                            legend_overlay.Draw()

                            # Almacenar referencias para evitar recolección de basura
                            setattr(self, f"overlay_obj_{num_type}_{den_type}_{current_particle}_{plot_type}", h_num_overlay)
                            setattr(self, f"overlay_obj_den_{num_type}_{den_type}_{current_particle}_{plot_type}", h_den_overlay)
                            setattr(self, f"overlay_legend_{num_type}_{den_type}_{current_particle}_{plot_type}", legend_overlay)
                        else:
                            print(f"Faltan datos o division por cero para overlay {num_type}/{den_type} de {plot_type} ({current_particle})")

                        pad_counter += 1
                        
                # Guardar ambos canvases para esta partícula
                canvas_ratios.Draw()
                canvas_ratios.SaveAs(str(self.BASE_DIR / "img" / f"comparative_ratios_{current_particle}.png"))
                setattr(self, f"canvas_ratios_{current_particle}", canvas_ratios)

                canvas_overlay.Draw()
                canvas_overlay.SaveAs(str(self.BASE_DIR / "img" / f"comparative_overlay_{current_particle}.png"))
                setattr(self, f"canvas_overlay_{current_particle}", canvas_overlay)
        
        # =========================================================================
        # PARTE 3: PROTEGER REFERENCIAS EN MEMORIA (Para Jupyter)
        # =========================================================================
        # Almacenar todas las variables locales evita que Python las recolecte
        # como basura mientras se visualizan en Jupyter. ROOT necesita que los
        # objetos persistan en memoria.
        self._keep_alive = locals().copy()
    
    def free_memory(self):
        """Libera la memoria de objetos gráficos de ROOT y recolecta basura.
        
        Se llama al inicio de cada función de gráficos para limpiar residuos
        de ejecuciones anteriores. Esto previene acumulación de memoria.
        
        Proceso:
        1. Vaciar el diccionario de protección de referencias (_keep_alive)
        2. Cerrar canvases de ROOT explícitamente
        3. Ejecutar recolector de basura de Python (gc.collect)
        """
        # Paso 1: Limpiar el "escudo" de referencias.
        # Este diccionario protege objetos ROOT de ser recolectados prematuramente.
        if hasattr(self, '_keep_alive'):
            self._keep_alive.clear()
            del self._keep_alive

        # Paso 2: Cerrar todos los canvases ROOT creados por la clase.
        # ROOT gestiona su propia memoria, así que conviene cerrar explícitamente cada lienzo.
        if hasattr(self, 'canvas_comparasion') and self.canvas_comparasion:
            self.canvas_comparasion.Close()
            self.canvas_comparasion = None

        if hasattr(self, 'canvas_mass_invariant') and self.canvas_mass_invariant:
            self.canvas_mass_invariant.Close()
            self.canvas_mass_invariant = None

        if hasattr(self, 'canvas_invariant_mass_overlay_channels') and self.canvas_invariant_mass_overlay_channels:
            self.canvas_invariant_mass_overlay_channels.Close()
            self.canvas_invariant_mass_overlay_channels = None

        if hasattr(self, 'canvas_invariant_mass_overlay_types_by_channel') and self.canvas_invariant_mass_overlay_types_by_channel:
            self.canvas_invariant_mass_overlay_types_by_channel.Close()
            self.canvas_invariant_mass_overlay_types_by_channel = None

        # Paso 3: Ejecutar recolector de basura de Python para liberar memoria.
        import gc
        gc.collect()
        
        print("Memoria gráfica liberada exitosamente.")



