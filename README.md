# Mantu Light Mesh

A WebGL-powered mesh gradient generator designed for creating "eclipse-style" lighting effects using the Mantu brand color palette.

## 🌟 Features

-   **Eclipse-Style Lighting**: Solid shapes (Circle, Square, Diamond, Mantu Logo) create unique lighting interactions where light wraps around occluding objects.
-   **Mantu Identity**: Built-in access to the official Mantu brand palette (Mind, Core, Soma, Aura).
-   **Interactive Canvas**:
    -   **Click & Drag** to position light sources.
    -   **Double-click** points to cycle through colors.
    -   **Real-time** property editing (Intensity, Radius, Softness, Stretch, Angle).
-   **Advanced Filters**: Apply global effects like Grain, Bloom, Chromatic Aberration, and Scanlines.
-   **Export Ready**: Download your creation as a high-res **PNG** or generate CSS for web implementation (approximation).

## 🚀 Getting Started

### Prerequisites

-   Node.js (v14 or higher)
-   npm or yarn

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/arthurhazan/mantu-light-mesh.git
    cd mantu-light-mesh
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Run the development server:
    ```bash
    npm run dev
    ```

4.  Open your browser and navigate to `http://localhost:5173`.

## 🛠️ Technologies

-   **React**: UI and State Management.
-   **WebGL**: Core rendering engine using custom Vertex and Fragment Shaders (GLSL).
-   **Vite**: Fast development build tool.

## 🎮 How to Use

1.  **Add Light**: Click anywhere on the black canvas to add a new light source.
2.  **Edit Light**: Click a light handle to open its properties panel. Change its shape, size, color, and intensity.
3.  **Shuffle**: Use the "Shuffle" button to randomize positions. Lock specific colors (🔓 → 🔒) to keep them in place while shuffling others.
4.  **Export**: Click "PNG" to save an image or "{ } CSS" to get the CSS code snippet.

## 📄 License

This project is for internal use and creative exploration.
