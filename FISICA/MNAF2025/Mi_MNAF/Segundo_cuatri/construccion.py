import ROOT as rt
from pathlib import Path
import os
import pandas as pd


class construccion_individually:
    """Construye estructuras de eventos y calcula la masa invariante del bosón Z.

    A partir de un archivo ``.dat`` con eventos tipo ``ee`` o ``mumu``, la clase:
    1. Identifica el canal leptónico según el nombre del archivo.
    2. Extrae los 4-momentos de ambas partículas por evento.
    3. Guarda variables cinemáticas de cada partícula en un diccionario.
    4. Calcula la masa invariante por evento del sistema de dos leptones.

    Attributes:
        path (Path): Ruta del archivo de entrada.
        events (dict): Información cinemática por partícula y por evento.
        invariant_mass_Z (dict): Masa invariante por índice de evento.
        events_df (pd.DataFrame): Tabla resumen exportada a CSV con ambas partículas.
    """

    def __init__(self, path, type="gen"):
        """Inicializa la construcción de eventos y su masa invariante.
        
        Workflow completo:
        1. Lee el archivo .dat y clasifica el canal leptónico
        2. Extrae 4-momentos y calcula variables cinemáticas
        3. Calcula masa invariante del sistema dileptónico
        4. Exporta resultados a CSV para análisis posterior
        5. Prepara directorios de salida

        Args:
            path (str | Path): Ruta al archivo de datos ``.dat``.
            type (str): Tipo de eventos a procesar (gen, reco, data).
        """
        # Configurar rutas base y archivo de entrada
        self.BASE_DIR = Path(__file__).resolve().parent
        self.path = Path(path)
        self.type = type
        
        # PASO 1: Extraer eventos y clasificar canal
        self.events = self._selection()
        
        # PASO 2: Calcular masa invariante y variables del sistema
        self.invariant_mass_Z = self._invariant_mass_and_variables()
        
        # PASO 3: Crear directorios de salida si no existen
        os.makedirs(self.BASE_DIR / "img", exist_ok=True)
        os.makedirs(self.BASE_DIR / "results", exist_ok=True)
        self.events_df = pd.DataFrame(self.events[self.p1_name]).T.join(
            pd.DataFrame(self.events[self.p2_name]).T,
            lsuffix=f"_{self.p1_name}",
            rsuffix=f"_{self.p2_name}",
        )
        self.events_df["invariant_mass_Z"] = self.invariant_mass_Z.values()
        self.events_df["eta_canal"] = self.eta_canal.values()
        self.events_df["pt_canal"] = self.pt_canal.values()
        self.events_df.to_csv(
            self.BASE_DIR
            / "results"
            / f"events_data_{self.type}_{self.p1_name}_{self.p2_name}.csv",
            index_label="event_index",
        )

    def _selection(self):
        """Lee el archivo, clasifica el canal leptónico y extrae 4-momentos por partícula.

        **Workflow de extracción:**

        1. Detecta canal (ee o μμ) del nombre del archivo usando glob matching
        2. Inicializa diccionarios para ambas partículas del par leptónico
        3. Lee .dat saltando las dos primeras líneas de metadatos
        4. Para cada evento: crea TLorentzVector y extrae variables cinemáticas
        5. Almacena px, py, pz, E (y derivadas: pT, η, masa, φ) por evento

        **Formato de archivo esperado** (.dat):
        - Línea 1: Metadatos/cabecera (ignorada)
        - Línea 2: Metadatos/cabecera (ignorada)
        - Línea 3+: [evt_id, dummy, px₁, py₁, pz₁, E₁, px₂, py₂, pz₂, E₂, ...]

        Returns:
            dict: Diccionario anidado ``{particle_name: {event_id: {variable: value, ...}, ...}, ...}``
                  donde particle_name ∈ {electron, positron, muon, anti-muon}
                  y variables = {px, py, pz, pt, eta, mass, phi, energy}

        Raises:
            ValueError: Si el nombre del archivo no contiene '_ee_' o '_mumu_'.

        Example:
            >>> obj._selection()  # Si path es 'Data/*_ee_*.dat'
            >>> obj.events['electron'][1]  # Evento 1, primera partícula
            {'px': -5.23, 'py': 12.15, 'pz': 45.8, 'pt': 13.1, 'eta': 1.25, ...}
        """
        self.events = {}
        
        # PASO 1: Detectar canal leptónico a partir del nombre del archivo
        if self.path.match("**/*_ee_*.dat"):
            self.p1_name = "electron"
            self.p2_name = "positron"
        elif self.path.match("**/*_mumu_*.dat"):
            self.p1_name = "muon"
            self.p2_name = "anti-muon"
        else:
            raise ValueError(f"Archivo '{self.path.name}' no contiene '_ee_' ni '_mumu_' en el nombre.")
        
        # PASO 2: Inicializar estructuras de datos para ambas partículas
        self.events[self.p1_name] = {}
        self.events[self.p2_name] = {}
        
        # PASO 3: Leer archivo .dat, omitiendo cabeceras iniciales
        with open(self.path, "r") as f:
            next(f)  # Omit line 1: metadata
            next(f)  # Omit line 2: metadata
            data = f.read().splitlines()
        
        # PASO 4: Procesar cada evento y extraer 4-momentos
        for i, line in enumerate(data):
            line = line.split()
            
            # Reconstruir TLorentzVector para partícula 1
            # (línea[2:6] = [px1, py1, pz1, E1])
            p1 = rt.TLorentzVector()
            p1.SetPxPyPzE(float(line[2]), float(line[3]), float(line[4]), float(line[5]))
            
            # Reconstruir TLorentzVector para partícula 2
            # (línea[6:10] = [px2, py2, pz2, E2])
            p2 = rt.TLorentzVector()
            p2.SetPxPyPzE(float(line[6]), float(line[7]), float(line[8]), float(line[9]))
            
            # PASO 5: Extraer magnitudes cinemáticas derivadas de cada 4-vector
            # Evento i+1 (indices empiezan en 1, no en 0)
            self.events[self.p1_name][i + 1] = {
                "px": p1.Px(),        # Momento en x (GeV/c)
                "py": p1.Py(),        # Momento en y (GeV/c)
                "pz": p1.Pz(),        # Momento en z (GeV/c)
                "pt": p1.Pt(),        # Momento transversal = √(px² + py²) (GeV/c)
                "eta": p1.Eta(),      # Pseudorapidez = -ln[tan(θ/2)]
                "mass": p1.M(),       # Masa invariante (GeV/c²) ≈ 0 para leptones
                "phi": p1.Phi(),      # Ángulo azimutal en plano xy (rad)
                "energy": p1.Energy(),# Energía E (GeV)
            }
            
            self.events[self.p2_name][i + 1] = {
                "px": p2.Px(),
                "py": p2.Py(),
                "pz": p2.Pz(),
                "pt": p2.Pt(),
                "eta": p2.Eta(),
                "mass": p2.M(),
                "phi": p2.Phi(),
                "energy": p2.Energy(),
            }
        
        return self.events

    def _invariant_mass_and_variables(self):
        """Calcula masa invariante del sistema dileptónico y variables del par.

        **Física:**
        
        La masa invariante del par leptónico es el invariante de Lorentz del 4-vector suma:
        
        m² = (E₁ + E₂)² - |p₁ + p₂|²  (en unidades naturales c=1)
        
        Se calcula como (p₁ + p₂).M() mediante la librería ROOT. En decaimiento Z→ll,
        la masa invariante reconstruida debe centrar alrededor de m_Z ≈ 91.2 GeV/c².
        
        El método también calcula pseudorapidez y pT del par (centro de masa dileptónico).

        **Workflow:**
        
        1. Itera sobre todos los eventos almacenados en self.events
        2. Reconstruye 4-vectores p1, p2 a partir de componentes guardados
        3. Calcula masa invariante como (p1 + p2).M()
        4. Calcula pseudorapidez del par η_ll = (p1 + p2).Eta()
        5. Calcula pT del par pT_ll = (p1 + p2).Pt()
        6. Almacena en diccionarios indexados por event_id

        Returns:
            dict: ``{event_id: invariant_mass_value, ...}`` con M en GeV/c².
                  Nota: self.eta_canal y self.pt_canal también se rellenan.

        Example:
            >>> obj._invariant_mass_and_variables()
            >>> obj.invariant_mass_Z[1]  # Masa invariante evento 1 (≈ 91 GeV si Z)
            91.2345
            >>> obj.eta_canal[1]  # Pseudorapidez del par
            0.5123
        """
        self.invariant_mass_Z = {}
        self.eta_canal = {}
        self.pt_canal = {}
        
        # Iterar sobre todos los eventos procesados en _selection()
        for i in self.events[self.p1_name]:
            # PASO 1: Reconstruir 4-vector de la primera partícula (e⁻ o μ⁻)
            p1 = rt.TLorentzVector()
            p1.SetPxPyPzE(
                self.events[self.p1_name][i]["px"],
                self.events[self.p1_name][i]["py"],
                self.events[self.p1_name][i]["pz"],
                self.events[self.p1_name][i]["energy"],
            )
            
            # PASO 2: Reconstruir 4-vector de la segunda partícula (e⁺ o μ⁺)
            p2 = rt.TLorentzVector()
            p2.SetPxPyPzE(
                self.events[self.p2_name][i]["px"],
                self.events[self.p2_name][i]["py"],
                self.events[self.p2_name][i]["pz"],
                self.events[self.p2_name][i]["energy"],
            )
            
            # PASO 3: Calcular masa invariante del par
            # M² = (E₁+E₂)² - |p₁+p₂|² via 4-vector suma
            invariant_mass = (p1 + p2).M()
            
            # PASO 4: Calcular pseudorapidez del sistema dileptónico
            eta_canal = (p1 + p2).Eta()
            
            # PASO 5: Calcular momento transversal del sistema
            pt_canal = (p1 + p2).Pt()
            
            # PASO 6: Guardar resultados indexados por event_id
            self.eta_canal[i] = eta_canal
            self.pt_canal[i] = pt_canal
            self.invariant_mass_Z[i] = invariant_mass
        
        return self.invariant_mass_Z
    
    def apply_mass_fit(self, hist, fit_type="gaus"):
        """Aplica ajuste matemático al histograma de masa invariante.

        **Tipos de ajuste disponibles:**

        1. **gaus** (Gaussiana): Función de distribución normal simétrica.
           Parámetros: [0]=Amplitud, [1]=Media (μ), [2]=Desv.Est. (σ)
           Rango: media ± 15 GeV/c²

        2. **bw** (Breit-Wigner): Forma resonante de partículas inestables.
           Parámetros: [0]=Normalización, [1]=Masa resonante, [2]=Ancho natural (Γ)
           Modela picos de resonancia con colas asimétricas.

        3. **voigt** (Perfil de Voigt): Convolución de BW y Gaussiana.
           Parámetros: [0]=Norm, [1]=Media, [2]=σ_detector, [3]=Γ_física
           Combina resolución experimental (Gauss) con resolución teórica (BW).

        4. **cruijff** (Cruijff empírico): Función asimétrica personalizada.
           Parámetros: [0]=Norm, [1]=Media, [2-3]=σ (izq/der), [4-5]=α (izq/der)
           Flexible para formas asimétricas, common in particle physics.

        **Workflow:**

        1. Limpia memoria de ROOT (free_memory)
        2. Detecta centro del pico automáticamente (bin con máximo contenido)
        3. Configura función con parámetros iniciales
        4. Realiza ajuste en rango: media ± 15 GeV/c²
        5. Calcula χ²/NDF y imprime resultado
        6. Retorna función ajustada para ploteo

        Args:
            hist (TH1F): Histograma a ajustar.
            fit_type (str): Tipo de ajuste ("gaus", "bw", "voigt", "cruijff").

        Returns:
            TF1: Función ajustada (puede ser None si tipo no reconocido).

        Example:
            >>> func = obj.apply_mass_fit(hist_mass, "voigt")
            >>> hist_mass.Fit(func, "...")  # ya hecho dentro del método
        """
        self.free_memory()  # Liberar memoria de ROOT antes de crear nuevos objetos gráficos
        
        # PASO 1: Detectar automáticamente el centro del pico
        max_bin = hist.GetMaximumBin()
        peak_x = hist.GetXaxis().GetBinCenter(max_bin)
        max_y = hist.GetMaximum()

        # PASO 2: Definir rango de ajuste (± 15 GeV alrededor del pico)
        fit_min = peak_x - 15.0
        fit_max = peak_x + 15.0

        # PASO 3: Crear función según tipo de ajuste
        if fit_type == "gaus":
            # Función nativa de ROOT
            func = rt.TF1(f"fit_gaus_{hist.GetName()}", "gaus", fit_min, fit_max)
            func.SetParameters(max_y, peak_x, 2.5)  # Estimaciones iniciales
            func.SetLineColor(rt.kRed)

        elif fit_type == "bw":
            # Breit-Wigner: función TMath de ROOT
            # [0]=Normalización, [1]=Media (masa), [2]=Gamma (ancho)
            func = rt.TF1(f"fit_bw_{hist.GetName()}", "[0]*TMath::BreitWigner(x, [1], [2])", fit_min, fit_max)
            func.SetParameters(max_y * 10, peak_x, 2.5)
            func.SetLineColor(rt.kOrange + 1)

        elif fit_type == "voigt":
            # Voigt: convolución de Breit-Wigner + Gaussiana
            # [0]=Norm, [1]=Media, [2]=Sigma_detector, [3]=Gamma_física
            func = rt.TF1(f"fit_voigt_{hist.GetName()}", "[0]*TMath::Voigt(x - [1], [2], [3])", fit_min, fit_max)
            func.SetParameters(max_y * 10, peak_x, 1.5, 2.5) 
            func.SetLineColor(rt.kMagenta + 1)

        elif fit_type == "cruijff":
            # Función Cruijff: forma asimétrica personalizada
            # Necesita inyección de código C++ en ROOT
            if not hasattr(self, "_cruijff_defined"):
                rt.gInterpreter.Declare("""
                double cruijff(double *x, double *par) {
                    // par[0]=Norm, par[1]=Media, par[2]=SigmaL, par[3]=SigmaR, par[4]=AlphaL, par[5]=AlphaR
                    double dx = x[0] - par[1];
                    double sigma = (dx < 0) ? par[2] : par[3];
                    double alpha = (dx < 0) ? par[4] : par[5];
                    double f = 2 * sigma * sigma + alpha * dx * dx;
                    // Protección: evitar divisiones por cero y overflow exponencial
                    if (f <= 0.0) return 0.0; 
                    return par[0] * exp(-dx*dx / f);
                }
                """)
                self._cruijff_defined = True

            func = rt.TF1(f"fit_cruijff_{hist.GetName()}", rt.cruijff, fit_min, fit_max, 6)
            func.SetParNames("Norm", "Mean", "SigmaL", "SigmaR", "AlphaL", "AlphaR")
            
            # Semillas iniciales
            func.SetParameters(max_y, peak_x, 2.0, 2.0, 0.1, 0.1)
            
            # Límites de seguridad
            func.SetParLimits(2, 0.1, 10.0)  # SigmaL > 0
            func.SetParLimits(3, 0.1, 10.0)  # SigmaR > 0
            func.SetParLimits(4, 0.0, 5.0)   # AlphaL >= 0
            func.SetParLimits(5, 0.0, 5.0)   # AlphaR >= 0
            
            func.SetLineColor(rt.kAzure + 2)

        else:
            print(f"Tipo de ajuste '{fit_type}' no reconocido. Opciones: gaus, bw, voigt, cruijff")
            return None

        # PASO 4: Configuración de línea y realización del ajuste
        func.SetLineWidth(3)
        func.SetLineStyle(1)

        # Realizar ajuste: R=respeta rango, S=guarda datos, +=superposición, Q=silencioso
        hist.Fit(func, "R S + Q")
        
        # PASO 5: Calcular χ² reducido y mostrar resultado
        chi2 = func.GetChisquare()
        ndf = func.GetNDF()
        
        if ndf > 0:
            chi2_reducido = chi2 / ndf
            nombre_particulas = f"{self.p1_name}-{self.p2_name}"
            print(f"Ajuste [{fit_type.upper():<7}] en {nombre_particulas:<18} | Chi2/NDF: {chi2:>7.2f} / {ndf:<3} = {chi2_reducido:.3f}")
        else:
            print(f"Ajuste [{fit_type.upper():<7}] en {self.p1_name}-{self.p2_name} | ERROR: NDF es 0.")
        
        return func

    def plotter(self):
        """Genera visualizaciones comparativas de variables cinemáticas e histograma de masa invariante.

        **Arquitectura de salida (2 Canvas):**

        1. **canvas_comparasion** (2×4 layout): Comparación pT y η para ambas partículas del par
           - Pad (1): Histograma individual pT particle1
           - Pad (2): Histograma individual η particle1
           - Pad (3): Histograma individual pT particle2
           - Pad (4): Histograma individual η particle2
           - Pad (5): Overlay normalizado pT (p1 vs p2)
           - Pad (6): Overlay normalizado η (p1 vs p2)
           - Pad (7): Ratio plot pT con línea de referencia y=1
           - Pad (8): Ratio plot η con línea de referencia y=1

        2. **canvas_mass_invariant** (1×1): Histograma de masa invariante dileptónica
           - Rango: 0 a 500 GeV/c²
           - Bins: 200
           - Pico esperado: m_Z ≈ 91.2 GeV/c² si eventos reales

        **Funciones auxiliares internas:**
        - _bin_range_with_content: Encuentra primer/último bin con contenido
        - _quantile_bin_range: Ajusta rango para contener fracción central de eventos (default 99.5%)
        - _autofit_single_hist: Ajusta ejes x/y para histograma individual
        - _autofit_hist_pair: Ajusta ejes x/y comunes para dos histogramas overlaid
        - _autofit_ratio_hist: Ajusta ejes para plots de ratio (ancla y=1)
        - _normalize_hist: Normaliza histograma al área = 1

        **Salida:**
        - Guarda PNG en img/ directorio:
          * ``{type}_comparasion_pt_eta_{p1_name}_{p2_name}.png``
          * ``{type}_invariant_mass_Z_{p1_name}_{p2_name}.png``
        - Almacena canvas en self.canvas_comparasion, self.canvas_mass_invariant

        **Physics:**
        - pT = momento transversal = √(px² + py²) [GeV/c]
        - η = pseudorapidez = -ln[tan(θ/2)] (medida de ángulo polar)
        - m_ll = masa invariante dileptónica
        """
        # PASO 0: Liberar memoria de ROOT antes de crear nuevos objetos gráficos
        self.free_memory()
        
        # ========== FUNCIONES AUXILIARES (definidas localmente dentro de plotter) =========
        
        def _bin_range_with_content(hist, include_errors=False):
            """Encuentra primer y último bin que contiene contenido no nulo.

            Útil para determinar el rango dinámico de un histograma cuando 
            hay regiones vacías en los extremos.

            Args:
                hist (TH1F): Histograma ROOT.
                include_errors (bool): Si True, incluye error estadístico al 
                    decidir si bin está "vacío".

            Returns:
                tuple: (first_bin, last_bin, x_min, x_max) en unidades de bins 
                    y ejes físicos, o None si histograma completamente vacío.
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
            x_min = hist.GetXaxis().GetBinLowEdge(first_bin)
            x_max = hist.GetXaxis().GetBinUpEdge(last_bin)
            return first_bin, last_bin, x_min, x_max

        def _quantile_bin_range(hist, central_fraction=0.995):
            """Calcula rango de bins que contiene fracción central de eventos.

            **Algoritmo:**

            1. Calcula integral total del histograma
            2. Determina tamaño de cada cola = (1 - central_fraction) / 2
            3. Busca desde el lado izquierdo hasta acumular tail_events
            4. Busca desde el lado derecho hasta acumular tail_events
            5. Retorna rango en bins y coordenadas físicas

            Este método es robusto para distribuciones asimétricas y elimina 
            automáticamente regiones de bajo contenido en los extremos.

            Args:
                hist (TH1F): Histograma ROOT.
                central_fraction (float): Fracción central a retener (default 0.995 = 99.5%).

            Returns:
                tuple: (low_bin, high_bin, x_min, x_max) con rango que contiene 
                    central_fraction del total, o (1, n_bins, x_min, x_max) si integral ≤ 0.
            """
            n_bins = hist.GetNbinsX()
            total = hist.Integral(1, n_bins)
            if total <= 0.0:
                return _bin_range_with_content(hist)

            # Calcular fracción de eventos en cada cola
            tail_fraction = max(0.0, min(0.5, (1.0 - central_fraction) / 2.0))
            tail_events = total * tail_fraction

            # Búsqueda desde izquierda hasta acumular tail_events
            low_bin = 1
            cumulative = 0.0
            while low_bin <= n_bins:
                next_cumulative = cumulative + hist.GetBinContent(low_bin)
                if next_cumulative > tail_events:
                    break
                cumulative = next_cumulative
                low_bin += 1

            # Búsqueda desde derecha hasta acumular tail_events
            high_bin = n_bins
            cumulative = 0.0
            while high_bin >= 1:
                next_cumulative = cumulative + hist.GetBinContent(high_bin)
                if next_cumulative > tail_events:
                    break
                cumulative = next_cumulative
                high_bin -= 1

            # Validación de rango
            low_bin = max(1, min(low_bin, n_bins))
            high_bin = max(1, min(high_bin, n_bins))
            if low_bin > high_bin:
                low_bin, high_bin = 1, n_bins

            x_min = hist.GetXaxis().GetBinLowEdge(low_bin)
            x_max = hist.GetXaxis().GetBinUpEdge(high_bin)
            return low_bin, high_bin, x_min, x_max

        def _autofit_single_hist(hist, y_padding=1.2, include_errors=False, anchor_y_one=False, central_fraction=0.995):
            """Ajusta automáticamente los rangos de ejes x/y para un histograma individual.

            **Workflow:**

            1. Usa _quantile_bin_range para identificar región útil en eje x
            2. Busca máximo contenido (± error si include_errors=True) en región
            3. Aplica padding multiplicativo al eje y para margen visual
            4. Si anchor_y_one=True, asegura que y_max >= 1 (útil para ratios)

            Args:
                hist (TH1F): Histograma a ajustar.
                y_padding (float): Factor multiplicador para margen en y (default 1.2 = 20% extra).
                include_errors (bool): Si True, busca máximo incluyendo barras de error.
                anchor_y_one (bool): Si True, asegura y_max >= 1 (para plots de ratio).
                central_fraction (float): Fracción central a contener (default 0.995).

            Returns:
                None. Modifica hist in-place vía SetRangeUser(), SetMinimum(), SetMaximum().
            """
            range_info = _quantile_bin_range(hist, central_fraction=central_fraction)
            if range_info is None:
                raise ValueError("No se encontraron bins con contenido en el histograma.")
            first_bin, last_bin, x_min, x_max = range_info
            hist.GetXaxis().SetRangeUser(x_min, x_max)
            
            # Encontrar máximo contenido en región útil
            y_max = 0.0
            for bin_idx in range(first_bin, last_bin + 1):
                content = hist.GetBinContent(bin_idx)
                error = hist.GetBinError(bin_idx) if include_errors else 0.0
                y_max = max(y_max, content + error)
            
            # Anclar a y=1 si es plot de ratio
            if anchor_y_one:
                y_max = max(y_max, 1.0)
            
            # Aplicar padding y establecer límites
            hist.SetMinimum(0.0)
            hist.SetMaximum(max(1e-6, y_max * y_padding))
            return None

        def _autofit_hist_pair(hist1, hist2, y_padding=1.2, central_fraction=0.995):
            """Ajusta rango común para dos histogramas superpuestos.

            **Workflow:**

            1. Calcula rango x útil para hist1 y hist2 por separado
            2. Toma unión de ambos rangos para eje x compartido
            3. Encuentra máximo contenido de ambos histogramas
            4. Aplica padding multiplicativo común

            Usado para plots overlay donde se necesita escala visual coherente.

            Args:
                hist1 (TH1F): Primer histograma.
                hist2 (TH1F): Segundo histograma.
                y_padding (float): Factor multiplicador para margen en y (default 1.2).
                central_fraction (float): Fracción central a contener (default 0.995).

            Returns:
                None. Modifica hist1 y hist2 in-place.
            """
            range1 = _quantile_bin_range(hist1, central_fraction=central_fraction)
            range2 = _quantile_bin_range(hist2, central_fraction=central_fraction)
            x_candidates = []
            if range1 is not None:
                x_candidates.append((range1[2], range1[3]))
            if range2 is not None:
                x_candidates.append((range2[2], range2[3]))
            if x_candidates:
                x_min = min(val[0] for val in x_candidates)
                x_max = max(val[1] for val in x_candidates)
                hist1.GetXaxis().SetRangeUser(x_min, x_max)
                hist2.GetXaxis().SetRangeUser(x_min, x_max)
            hist1.SetMinimum(0.0)
            hist2.SetMinimum(0.0)
            hist1.SetMaximum(max(hist1.GetMaximum(), hist2.GetMaximum()) * y_padding)
            return None

        def _autofit_ratio_hist(hist, y_padding=1.25, central_fraction=0.995):
            """Ajusta ejes para histograma de ratio con referencia anclada en y=1.

            **Workflow para ratio plots:**

            1. Calcula rango x útil usando _quantile_bin_range
            2. Encuentra mín y máx de contenido ± error en región
            3. Asegura que y=1 está dentro del rango (referencia central)
            4. Aplica padding simétrico alrededor de y=1

            Args:
                hist (TH1F): Histograma de ratio (esperado ratio data/MC o similar).
                y_padding (float): Factor multiplicador para margen (default 1.25 = 25% extra).
                central_fraction (float): Fracción central a contener (default 0.995).

            Returns:
                None. Modifica hist in-place vía SetRangeUser() en eje y.
            """
            range_info = _quantile_bin_range(hist, central_fraction=central_fraction)
            if range_info is None:
                hist.GetYaxis().SetRangeUser(0.5, 1.5)
                return None

            first_bin, last_bin, x_min, x_max = range_info
            hist.GetXaxis().SetRangeUser(x_min, x_max)

            # Buscar mín y máx del ratio en región útil (incluir errores)
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

            # Asegurar que y=1 está en el rango (línea de referencia)
            y_min = min(y_min, 1.0)
            y_max = max(y_max, 1.0)
            
            # Aplicar padding simétrico
            span = max(1e-6, y_max - y_min)
            margin = span * (y_padding - 1.0)
            hist.GetYaxis().SetRangeUser(y_min - margin, y_max + margin)
            return None

        def _normalize_hist(hist):
            """Normaliza histograma al área total = 1 para comparar formas.

            Util para overlay de dos muestras con diferente número de eventos.
            Después de normalización, el histograma es una distribución de probabilidad.

            Args:
                hist (TH1F): Histograma a normalizar (in-place modification).

            Returns:
                float: Integral original antes de normalización (para referencia).
            """
            total = hist.Integral()
            if total > 0.0:
                hist.Scale(1.0 / total)
            return total

        # ========== CANVAS 1: COMPARACIÓN pT y η (2×4 layout) =========
        canvas_comparasion = rt.TCanvas(
            "canvas_comparasion", "p_{T} y #eta generacion electron y positron", 1600, 1200
        )
        canvas_comparasion.Divide(2, 4)  # 2 columnas (pT, η) × 4 filas (p1 individual, p2 individual, overlay, ratio)
        
        # PAD 1: Histograma individual de pT para particle1
        canvas_comparasion.cd(1)
        hist_pt_p1 = rt.TH1F(
            f"hist_pt_{self.p1_name}",
            f"p_{{T}} de {self.p1_name};p_{{T}} (GeV/c);Eventos",
            200,  # 200 bins
            0,    # x_min = 0 GeV/c
            500,  # x_max = 500 GeV/c
        )
        # Llenar histograma con valores de pT de todos los eventos
        for pt in self.events[self.p1_name].values():
            hist_pt_p1.Fill(pt["pt"])
        hist_pt_p1.SetLineColor(rt.kAzure + 1)
        hist_pt_p1.SetFillColor(rt.kAzure - 9)
        _normalize_hist(hist_pt_p1)
        hist_pt_p1.GetYaxis().SetTitle("Eventos normalizados")
        _autofit_single_hist(hist_pt_p1)
        hist_pt_p1.Draw("HIST")
        
        # PAD 3: Histograma individual de pT para particle2
        canvas_comparasion.cd(3)
        hist_pt_p2 = rt.TH1F(
            f"hist_pt_{self.p2_name}",
            f"p_{{T}} de {self.p2_name};p_{{T}} (GeV/c);Eventos",
            200,
            0,
            500,
        )
        for pt in self.events[self.p2_name].values():
            hist_pt_p2.Fill(pt["pt"])
        hist_pt_p2.SetLineColor(rt.kOrange + 7)
        hist_pt_p2.SetFillColor(rt.kOrange - 3)
        _normalize_hist(hist_pt_p2)
        hist_pt_p2.GetYaxis().SetTitle("Eventos normalizados")
        _autofit_single_hist(hist_pt_p2)
        hist_pt_p2.Draw("HIST")
        
        # PAD 2: Histograma individual de η para particle1
        canvas_comparasion.cd(2)
        hist_eta_p1 = rt.TH1F(
            f"hist_eta_{self.p1_name}",
            f"#eta de {self.p1_name};#eta;Eventos",
            100,   # 100 bins
            -20,   # x_min = -20 (pseudorapidez)
            20,    # x_max = 20
        )
        for eta in self.events[self.p1_name].values():
            hist_eta_p1.Fill(eta["eta"])
        hist_eta_p1.SetLineColor(rt.kAzure + 1)
        hist_eta_p1.SetFillColor(rt.kAzure - 9)
        _normalize_hist(hist_eta_p1)
        hist_eta_p1.GetYaxis().SetTitle("Eventos normalizados")
        _autofit_single_hist(hist_eta_p1)
        hist_eta_p1.Draw("HIST")
        
        # PAD 4: Histograma individual de η para particle2
        canvas_comparasion.cd(4)
        hist_eta_p2 = rt.TH1F(
            f"hist_eta_{self.p2_name}",
            f"#eta de {self.p2_name};#eta;Eventos",
            100,
            -20,
            20,
        )
        for eta in self.events[self.p2_name].values():
            hist_eta_p2.Fill(eta["eta"])
        hist_eta_p2.SetLineColor(rt.kOrange + 7)
        hist_eta_p2.SetFillColor(rt.kOrange - 3)
        _normalize_hist(hist_eta_p2)
        hist_eta_p2.GetYaxis().SetTitle("Eventos normalizados")
        _autofit_single_hist(hist_eta_p2)
        hist_eta_p2.Draw("HIST")
        # PAD 5: Overlay normalizado de pT (particle1 vs particle2)
        canvas_comparasion.cd(5)
        hist_pt_p1_copy = hist_pt_p1.Clone("hist_pt_p1_copy")
        hist_pt_p2_copy = hist_pt_p2.Clone("hist_pt_p2_copy")
        hist_pt_p1_copy.SetLineColor(rt.kAzure + 1)
        hist_pt_p1_copy.SetFillStyle(0)  # Sin relleno para transparencia
        hist_pt_p2_copy.SetLineColor(rt.kOrange + 7)
        hist_pt_p2_copy.SetFillStyle(0)
        _normalize_hist(hist_pt_p1_copy)
        _normalize_hist(hist_pt_p2_copy)
        hist_pt_p1_copy.SetTitle(f"Comparacion de p_{{T}} entre {self.p1_name} y {self.p2_name}")
        hist_pt_p1_copy.GetXaxis().SetTitle("p_{T} (GeV/c)")
        hist_pt_p1_copy.GetYaxis().SetTitle("Eventos normalizados")
        _autofit_hist_pair(hist_pt_p1_copy, hist_pt_p2_copy)
        hist_pt_p1_copy.SetStats(0)  # Quitar caja de estadísticas
        hist_pt_p1_copy.Draw("HIST")
        hist_pt_p2_copy.Draw("HIST SAME")  # Superponer sobre el mismo pad
        # Añadir leyenda identificando cada línea
        leyenda_pt = rt.TLegend(0.7, 0.9, 0.9, 0.75)
        leyenda_pt.AddEntry(hist_pt_p1_copy, f"{self.p1_name}", "l")
        leyenda_pt.AddEntry(hist_pt_p2_copy, f"{self.p2_name}", "l")
        leyenda_pt.Draw()
        
        # PAD 6: Overlay normalizado de η (particle1 vs particle2)
        canvas_comparasion.cd(6)
        hist_eta_p1_copy = hist_eta_p1.Clone("hist_eta_p1_copy")
        hist_eta_p2_copy = hist_eta_p2.Clone("hist_eta_p2_copy")
        hist_eta_p1_copy.SetLineColor(rt.kAzure + 1)
        hist_eta_p1_copy.SetFillStyle(0)
        hist_eta_p2_copy.SetLineColor(rt.kOrange + 7)
        hist_eta_p2_copy.SetFillStyle(0)
        _normalize_hist(hist_eta_p1_copy)
        _normalize_hist(hist_eta_p2_copy)
        hist_eta_p1_copy.SetTitle(f"Comparacion de #eta entre {self.p1_name} y {self.p2_name}")
        hist_eta_p1_copy.GetXaxis().SetTitle("#eta")
        hist_eta_p1_copy.GetYaxis().SetTitle("Eventos normalizados")
        _autofit_hist_pair(hist_eta_p1_copy, hist_eta_p2_copy)
        hist_eta_p1_copy.SetStats(0)
        hist_eta_p1_copy.Draw("HIST")
        hist_eta_p2_copy.Draw("HIST SAME")
        leyenda_eta = rt.TLegend(0.7, 0.9, 0.9, 0.75)
        leyenda_eta.AddEntry(hist_eta_p1_copy, f"{self.p1_name}", "l")
        leyenda_eta.AddEntry(hist_eta_p2_copy, f"{self.p2_name}", "l")
        leyenda_eta.Draw()
        # PAD 7: Plot de ratio de pT con línea de referencia en y=1
        canvas_comparasion.cd(7)
        hist_ratio_pt_num = hist_pt_p1.Clone("h_ratio_pt_num")
        hist_ratio_pt_den = hist_pt_p2.Clone("h_ratio_pt_den")
        _normalize_hist(hist_ratio_pt_num)
        _normalize_hist(hist_ratio_pt_den)
        hist_ratio_pt = hist_ratio_pt_num.Clone("h_ratio_pt")
        hist_ratio_pt.Divide(hist_ratio_pt_den)  # Calcula ratio bin-by-bin con propagación de errores
        hist_ratio_pt.SetTitle(f"Razon de p_{{T}};p_{{T}} (GeV/c);{self.p1_name}/{self.p2_name}")
        hist_ratio_pt.SetYTitle(f"{self.p1_name}/{self.p2_name}")
        hist_ratio_pt.SetLineColor(rt.kBlack)
        hist_ratio_pt.SetMarkerStyle(20)
        hist_ratio_pt.SetMarkerSize(0.8)
        _autofit_ratio_hist(hist_ratio_pt)
        hist_ratio_pt.Draw("E")  # "E" = mostrar puntos con barras de error
        
        # Dibujar línea horizontal de referencia en y=1 (si ratio=1, ambas son iguales)
        x_first_pt = hist_ratio_pt.GetXaxis().GetFirst()
        x_last_pt = hist_ratio_pt.GetXaxis().GetLast()
        x_min_pt = hist_ratio_pt.GetXaxis().GetBinLowEdge(x_first_pt)
        x_max_pt = hist_ratio_pt.GetXaxis().GetBinUpEdge(x_last_pt)
        line_ratio = rt.TLine(x_min_pt, 1, x_max_pt, 1)
        line_ratio.SetLineColor(rt.kRed)
        line_ratio.SetLineStyle(2)  # Línea punteada
        line_ratio.Draw("same")
        
        # PAD 8: Plot de ratio de η con línea de referencia en y=1
        canvas_comparasion.cd(8)
        hist_ratio_eta_num = hist_eta_p1.Clone("hist_ratio_eta_num")
        hist_ratio_eta_den = hist_eta_p2.Clone("hist_ratio_eta_den")
        _normalize_hist(hist_ratio_eta_num)
        _normalize_hist(hist_ratio_eta_den)
        hist_ratio_eta = hist_ratio_eta_num.Clone("hist_ratio_eta")
        hist_ratio_eta.Divide(hist_ratio_eta_den)
        hist_ratio_eta.SetTitle(f"Razon de #eta;#eta;{self.p1_name}/{self.p2_name}")
        hist_ratio_eta.SetYTitle(f"{self.p1_name}/{self.p2_name}")
        hist_ratio_eta.SetLineColor(rt.kBlack)
        hist_ratio_eta.SetMarkerStyle(20)
        hist_ratio_eta.SetMarkerSize(0.8)
        _autofit_ratio_hist(hist_ratio_eta)
        hist_ratio_eta.Draw("E")
        
        # Línea de referencia en y=1
        x_first_eta = hist_ratio_eta.GetXaxis().GetFirst()
        x_last_eta = hist_ratio_eta.GetXaxis().GetLast()
        x_min_eta = hist_ratio_eta.GetXaxis().GetBinLowEdge(x_first_eta)
        x_max_eta = hist_ratio_eta.GetXaxis().GetBinUpEdge(x_last_eta)
        line_ratio_eta = rt.TLine(x_min_eta, 1, x_max_eta, 1)
        line_ratio_eta.SetLineColor(rt.kRed)
        line_ratio_eta.SetLineStyle(2)
        line_ratio_eta.Draw("same")
        
        # Guardar canvas de comparación y mantener referencia en memoria
        canvas_comparasion.Draw()
        canvas_comparasion.SaveAs(
            str.format(f"{self.BASE_DIR}/img/{self.type}_comparasion_pt_eta_{self.p1_name}_{self.p2_name}.png")
        )
        self.canvas_comparasion = canvas_comparasion
        # ========== CANVAS 2: MASA INVARIANTE DILEPTÓNICA =========
        # Histograma final de la masa invariante del sistema dileptónico
        hist = rt.TH1F(
            "hist",
            "Masa invariante del sistema dileptonico;m_{ll} (GeV/c^{2});Eventos",
            200,  # 200 bins
            0,    # x_min = 0 GeV/c²
            500,  # x_max = 500 GeV/c²
        )
        # Llenar histograma con todas las masas invariantes calculadas
        for mass in self.invariant_mass_Z.values():
            hist.Fill(mass)
        
        # Crear canvas dedicado para masa invariante
        canvas = rt.TCanvas(
            f"canvas_{self.p1_name}_{self.p2_name}_{self.type}", 
            "Masa invariante del sistema dileptonico", 
            800, 600
        )
        
        # Configuración visual
        hist.SetLineColor(rt.kTeal + 1)
        hist.SetFillColor(rt.kTeal - 8)
        _autofit_single_hist(hist)
        hist.Draw("HIST")
        
        # Guardar canvas de masa invariante
        canvas.Draw()
        canvas.SaveAs(
            str.format(f"{self.BASE_DIR}/img/{self.type}_invariant_mass_Z_{self.p1_name}_{self.p2_name}.png")
        )
        self.canvas_mass_invariant = canvas
        
        # Mantener referencias en memoria para evitar garbage collection
        self._keep_alive = locals().copy()
    def plot_mass_fits(self):
        """Genera gráfica comparativa de diferentes ajustes matemáticos al pico de masa.

        **Propósito:**

        Visualizar cómo diferentes funciones matemáticas (Gauss, Breit-Wigner, Voigt, Cruijff)
        describen el pico de masa invariante, permitiendo comparar χ²/NDF y formas de ajuste.

        **Workflow:**

        1. Crea histograma enfocado en región del pico (60-120 GeV/c², típico para Z)
        2. Llama apply_mass_fit() con cada tipo de ajuste (gaus, bw, voigt, cruijff)
        3. Superpone todas las funciones ajustadas sobre el histograma
        4. Construye leyenda y salva canvas

        **Salida:**
        - PNG en img/ directorio: ``{type}_mass_fits_{p1_name}_{p2_name}.png``
        - Canvas almacenado en self.canvas_mass_fits

        **Uso:**
        Llamar después de plotter() para análisis adicional de forma del pico.
        """
        # PASO 0: Liberar memoria de iteraciones anteriores
        self.free_memory()
        
        # PASO 1: Crear histograma enfocado en región del pico (60-120 GeV/c²)
        hist_name = f"hist_mass_fits_{self.type}_{self.p1_name}_{self.p2_name}"
        hist_fits = rt.TH1F(
            hist_name,
            f"Ajustes de Masa Invariante ({self.type});m_{{ll}} (GeV/c^{{2}});Eventos",
            100,   # 100 bins para resolución en pico
            60,    # x_min = 60 GeV/c² (por debajo del pico Z)
            120    # x_max = 120 GeV/c² (por encima del pico Z)
        )
        
        # Llenar histograma con masas invariantes en rango del pico
        for mass in self.invariant_mass_Z.values():
            hist_fits.Fill(mass)
            
        # PASO 2: Crear canvas dedicado para visualización de ajustes
        canvas_name = f"canvas_fits_{self.p1_name}_{self.p2_name}_{self.type}"
        canvas_fits = rt.TCanvas(canvas_name, "Comparativa de Ajustes", 800, 600)
        
        # Configuración visual del histograma de datos
        hist_fits.SetLineColor(rt.kBlack)
        hist_fits.SetMarkerStyle(20)     # Puntos
        hist_fits.SetMarkerSize(0.8)
        hist_fits.SetStats(0)             # Sin caja de estadísticas
        
        hist_fits.Draw("E")  # "E" = mostrar puntos con barras de error
        
        # PASO 3: Aplicar todos los tipos de ajuste y superponer curvas
        fit_gaus  = self.apply_mass_fit(hist_fits, "gaus")
        fit_bw    = self.apply_mass_fit(hist_fits, "bw")
        fit_voigt = self.apply_mass_fit(hist_fits, "voigt")
        fit_cruif = self.apply_mass_fit(hist_fits, "cruijff")
        
        # PASO 4: Construir leyenda identificando datos y cada ajuste
        leyenda = rt.TLegend(0.65, 0.65, 0.9, 0.85)
        leyenda.SetBorderSize(0)
        leyenda.AddEntry(hist_fits, f"Datos ({self.type})", "lep")
        if fit_gaus: 
            leyenda.AddEntry(fit_gaus, "Gaussiana", "l")
        if fit_bw: 
            leyenda.AddEntry(fit_bw, "Breit-Wigner", "l")
        if fit_voigt: 
            leyenda.AddEntry(fit_voigt, "Voigt (BW + Gauss)", "l")
        if fit_cruif:
            leyenda.AddEntry(fit_cruif, "Cruijff", "l")
        leyenda.Draw()
        
        # PASO 5: Guardar canvas de ajustes
        canvas_fits.Draw()
        nombre_archivo = f"{self.type}_mass_fits_{self.p1_name}_{self.p2_name}.png"
        canvas_fits.SaveAs(str(self.BASE_DIR / "img" / nombre_archivo))
        
        # PASO 6: Proteger objetos gráficos en memoria para Jupyter
        self.canvas_mass_fits = canvas_fits
        self.leyenda_mass_fits = leyenda
        
        # Actualizar escudo de memoria (no borra protecciones de plotter() si se llamó antes)
        if hasattr(self, '_keep_alive'):
            self._keep_alive.update(locals())
        else:
            self._keep_alive = locals().copy()
            
        print(f"Gráfica de ajustes generada exitosamente: {nombre_archivo}")

    def free_memory(self):
        """Libera memoria de objetos gráficos de ROOT y Python.

        **Problema que resuelve:**

        ROOT gestiona memoria en forma mixta (C++ + Python). Los objetos gráficos
        (TCanvas, TH1F) persisten en memoria del motor ROOT incluso después de ser
        anulados en Python, causando memory leaks especialmente en Jupyter notebooks.

        **Workflow de liberación (3 pasos):**

        1. **Limpiar diccionario protector (_keep_alive)**: Elimina referencias a
           objetos locales que Root mantiene vivos.

        2. **Cerrar objetos ROOT explícitamente**: Llama Close() en cada TCanvas
           y anula referencias Python.

        3. **Invocar garbage collector**: Fuerza limpieza inmediata mediante gc.collect()

        **Efectos:**
        - Libera memoria de canvases previos antes de crear nuevos
        - Evita conflictos de nombres en ROOT
        - Permite reutilizar nombre sa de histogramas
        - Crítico para Jupyter notebooks de larga duración

        **Uso típico:**
        Se llama automáticamente al inicio de plotter() y plot_mass_fits()
        para evitar saturación de memoria.
        """
        # PASO 1: Destruir el "escudo" que protegía objetos locales de plotter()
        if hasattr(self, '_keep_alive'):
            self._keep_alive.clear()
            del self._keep_alive
            
        # PASO 2: Cerrar explícitamente canvases en motor ROOT (C++)
        # Se cierran todos los canvases creados por plotter() y plot_mass_fits()
        
        if hasattr(self, 'canvas_comparasion') and self.canvas_comparasion:
            self.canvas_comparasion.Close()  # Cierra canvas en ROOT
            self.canvas_comparasion = None    # Anula referencia Python
            
        if hasattr(self, 'canvas_mass_invariant') and self.canvas_mass_invariant:
            self.canvas_mass_invariant.Close()
            self.canvas_mass_invariant = None
            
        if hasattr(self, 'canvas_mass_fits') and self.canvas_mass_fits:
            self.canvas_mass_fits.Close()
            self.canvas_mass_fits = None
        
        # PASO 2b: Limpiar referencias a leyendas y otros objetos gráficos
        if hasattr(self, 'leyenda_mass_fits'):
            self.leyenda_mass_fits = None
            
        # PASO 3: Invocar garbage collector Python para limpieza inmediata
        import gc
        gc.collect()
        
        print("Memoria gráfica liberada exitosamente.")
