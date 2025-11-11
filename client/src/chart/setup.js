import { Chart as ChartJS, CategoryScale, LinearScale, LogarithmicScale, BarElement, PointElement, Tooltip, Legend, BubbleController } from 'chart.js';
import { TreemapController, TreemapElement } from 'chartjs-chart-treemap';

// Register chart elements and plugins once for the app
ChartJS.register(CategoryScale, LinearScale, LogarithmicScale, BarElement, PointElement, Tooltip, Legend, BubbleController, TreemapController, TreemapElement);

export default ChartJS;
