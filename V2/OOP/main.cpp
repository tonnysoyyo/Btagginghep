#include "JetAnalyzer.cpp"

int main() {
    // Lista de archivos de entrada
    std::vector<std::string> inputFiles = {
        "../../rootfiles/ZTobbbar_pT_20.root"
    };

    // Crear instancia de JetAnalyzer
    JetAnalyzer analyzer(inputFiles);

    // Procesar eventos
    analyzer.LoopEvents();

    // Dibujar histogramas
    analyzer.DrawHistograms();

    // Guardar histogramas
    analyzer.SaveHistograms("plots");

    std::cout << "El análisis ha finalizado correctamente." << std::endl;

    return 0;
}