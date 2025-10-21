import React from 'react';
import { ArrowRight, FileText, Layers, Grid3x3, BarChart3, Target, Lightbulb } from 'lucide-react';

const ThesisDiagram = () => {
  return (
    <div className="w-full h-full bg-gradient-to-br from-slate-50 to-blue-50 p-8 overflow-auto">
      <div className="max-w-7xl mx-auto">
        {/* Title */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-slate-800 mb-2">
            Textual Clusterization using SOMs and Word Embeddings
          </h1>
          <p className="text-lg text-slate-600">X → Y Framework: From Raw Text to Clustered Visualization</p>
        </div>

        {/* Main Process Flow */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 items-center">
            
            {/* X - INPUT SECTION */}
            <div className="lg:col-span-2 bg-gradient-to-br from-blue-100 to-blue-50 rounded-lg p-6 border-2 border-blue-300">
              <div className="flex items-center gap-3 mb-4">
                <FileText className="w-8 h-8 text-blue-600" />
                <h2 className="text-2xl font-bold text-blue-900">X (Input)</h2>
              </div>
              
              <div className="space-y-4">
                <div className="bg-white rounded-lg p-4">
                  <h3 className="font-semibold text-slate-800 mb-2">Textual Dataset</h3>
                  <ul className="text-sm text-slate-600 space-y-1">
                    <li>• Documents/Articles</li>
                    <li>• Reviews/Comments</li>
                    <li>• News Articles</li>
                    <li>• Research Papers</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* PROCESSING PIPELINE */}
            <div className="lg:col-span-2 space-y-4">
              
              {/* Step 1: Word Embeddings */}
              <div className="bg-gradient-to-r from-purple-100 to-purple-50 rounded-lg p-4 border border-purple-300">
                <div className="flex items-center gap-2 mb-3">
                  <Layers className="w-6 h-6 text-purple-600" />
                  <h3 className="font-bold text-purple-900">Step 1: Word Embeddings</h3>
                </div>
                <div className="text-sm space-y-2">
                  <div className="bg-white rounded p-2">
                    <p className="font-semibold text-slate-700">Hyperparameters:</p>
                    <ul className="text-slate-600 text-xs mt-1">
                      <li>• Vector dimensions (50, 100, 300)</li>
                      <li>• Context window size</li>
                      <li>• Min word frequency</li>
                      <li>• Training epochs</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded p-2">
                    <p className="font-semibold text-slate-700">Techniques:</p>
                    <ul className="text-slate-600 text-xs mt-1">
                      <li>• Word2Vec (CBOW/Skip-gram)</li>
                      <li>• GloVe</li>
                      <li>• FastText</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center">
                <ArrowRight className="w-6 h-6 text-slate-400" />
              </div>

              {/* Step 2: Dimensionality Reduction */}
              <div className="bg-gradient-to-r from-amber-100 to-amber-50 rounded-lg p-4 border border-amber-300">
                <div className="flex items-center gap-2 mb-3">
                  <BarChart3 className="w-6 h-6 text-amber-600" />
                  <h3 className="font-bold text-amber-900">Step 2: Feature Reduction</h3>
                </div>
                <div className="text-sm space-y-2">
                  <div className="bg-white rounded p-2">
                    <p className="font-semibold text-slate-700">Methods:</p>
                    <ul className="text-slate-600 text-xs mt-1">
                      <li>• PCA (Principal Component Analysis)</li>
                      <li>• t-SNE</li>
                      <li>• Autoencoder</li>
                      <li>• Feature selection techniques</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center">
                <ArrowRight className="w-6 h-6 text-slate-400" />
              </div>

              {/* Step 3: SOM Training */}
              <div className="bg-gradient-to-r from-green-100 to-green-50 rounded-lg p-4 border border-green-300">
                <div className="flex items-center gap-2 mb-3">
                  <Grid3x3 className="w-6 h-6 text-green-600" />
                  <h3 className="font-bold text-green-900">Step 3: SOM Training</h3>
                </div>
                <div className="text-sm space-y-2">
                  <div className="bg-white rounded p-2">
                    <p className="font-semibold text-slate-700">Hyperparameters:</p>
                    <ul className="text-slate-600 text-xs mt-1">
                      <li>• Grid size (10x10, 20x20, 30x30)</li>
                      <li>• Learning rate (α)</li>
                      <li>• Neighborhood radius (σ)</li>
                      <li>• Training iterations</li>
                      <li>• Topology (hexagonal/rectangular)</li>
                      <li>• Distance metric (Euclidean/Manhattan)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Y - OUTPUT SECTION */}
            <div className="lg:col-span-1 bg-gradient-to-br from-emerald-100 to-emerald-50 rounded-lg p-6 border-2 border-emerald-300">
              <div className="flex items-center gap-3 mb-4">
                <Target className="w-8 h-8 text-emerald-600" />
                <h2 className="text-2xl font-bold text-emerald-900">Y (Output)</h2>
              </div>
              
              <div className="space-y-4">
                <div className="bg-white rounded-lg p-4">
                  <h3 className="font-semibold text-slate-800 mb-2">SOM Visualization</h3>
                  <ul className="text-sm text-slate-600 space-y-1">
                    <li>• Cluster map</li>
                    <li>• U-Matrix</li>
                    <li>• Component planes</li>
                    <li>• Hit histogram</li>
                  </ul>
                </div>
                
                <div className="bg-white rounded-lg p-4">
                  <h3 className="font-semibold text-slate-800 mb-2">Metrics</h3>
                  <ul className="text-sm text-slate-600 space-y-1">
                    <li>• Silhouette score</li>
                    <li>• Davies-Bouldin index</li>
                    <li>• Quantization error</li>
                    <li>• Topographic error</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Experimental Matrix */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-slate-800 mb-4 flex items-center gap-2">
            <BarChart3 className="w-7 h-7 text-indigo-600" />
            Experimental Configuration Matrix
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-indigo-100 text-indigo-900">
                  <th className="p-3 text-left">Experiment</th>
                  <th className="p-3 text-left">Embedding Type</th>
                  <th className="p-3 text-left">Vector Dim</th>
                  <th className="p-3 text-left">Reduction</th>
                  <th className="p-3 text-left">SOM Grid</th>
                  <th className="p-3 text-left">Learning Rate</th>
                  <th className="p-3 text-left">Iterations</th>
                </tr>
              </thead>
              <tbody className="text-slate-700">
                <tr className="border-b hover:bg-slate-50">
                  <td className="p-3">Baseline</td>
                  <td className="p-3">Word2Vec</td>
                  <td className="p-3">300</td>
                  <td className="p-3">None</td>
                  <td className="p-3">20×20</td>
                  <td className="p-3">0.5</td>
                  <td className="p-3">10000</td>
                </tr>
                <tr className="border-b hover:bg-slate-50">
                  <td className="p-3">Exp-1</td>
                  <td className="p-3">GloVe</td>
                  <td className="p-3">100</td>
                  <td className="p-3">PCA (50)</td>
                  <td className="p-3">15×15</td>
                  <td className="p-3">0.3</td>
                  <td className="p-3">15000</td>
                </tr>
                <tr className="border-b hover:bg-slate-50">
                  <td className="p-3">Exp-2</td>
                  <td className="p-3">FastText</td>
                  <td className="p-3">200</td>
                  <td className="p-3">t-SNE (30)</td>
                  <td className="p-3">25×25</td>
                  <td className="p-3">0.7</td>
                  <td className="p-3">12000</td>
                </tr>
                <tr className="hover:bg-slate-50">
                  <td className="p-3">Exp-N</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Applications Section */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-2">
            <Lightbulb className="w-7 h-7 text-yellow-600" />
            Real-World Applications & Uses
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gradient-to-br from-blue-50 to-white rounded-lg p-5 border border-blue-200">
              <h3 className="font-bold text-blue-900 mb-3 text-lg">Document Organization</h3>
              <ul className="space-y-2 text-slate-700">
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 mt-1">►</span>
                  <span>Automatic categorization of large document repositories</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 mt-1">►</span>
                  <span>Email classification and inbox management</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 mt-1">►</span>
                  <span>News article clustering by topic</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-white rounded-lg p-5 border border-purple-200">
              <h3 className="font-bold text-purple-900 mb-3 text-lg">Customer Intelligence</h3>
              <ul className="space-y-2 text-slate-700">
                <li className="flex items-start gap-2">
                  <span className="text-purple-600 mt-1">►</span>
                  <span>Customer review sentiment clustering</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600 mt-1">►</span>
                  <span>Product feedback categorization</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600 mt-1">►</span>
                  <span>Support ticket auto-routing</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-white rounded-lg p-5 border border-green-200">
              <h3 className="font-bold text-green-900 mb-3 text-lg">Research & Academia</h3>
              <ul className="space-y-2 text-slate-700">
                <li className="flex items-start gap-2">
                  <span className="text-green-600 mt-1">►</span>
                  <span>Scientific literature topic mapping</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600 mt-1">►</span>
                  <span>Research trend identification</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600 mt-1">►</span>
                  <span>Citation network analysis</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-orange-50 to-white rounded-lg p-5 border border-orange-200">
              <h3 className="font-bold text-orange-900 mb-3 text-lg">Content Discovery</h3>
              <ul className="space-y-2 text-slate-700">
                <li className="flex items-start gap-2">
                  <span className="text-orange-600 mt-1">►</span>
                  <span>Recommendation systems</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-600 mt-1">►</span>
                  <span>Similar content suggestions</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-600 mt-1">►</span>
                  <span>Content gap analysis</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-red-50 to-white rounded-lg p-5 border border-red-200">
              <h3 className="font-bold text-red-900 mb-3 text-lg">Social Media Analysis</h3>
              <ul className="space-y-2 text-slate-700">
                <li className="flex items-start gap-2">
                  <span className="text-red-600 mt-1">►</span>
                  <span>Trend detection and tracking</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600 mt-1">►</span>
                  <span>Community opinion clustering</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600 mt-1">►</span>
                  <span>Brand monitoring and perception</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-indigo-50 to-white rounded-lg p-5 border border-indigo-200">
              <h3 className="font-bold text-indigo-900 mb-3 text-lg">Business Intelligence</h3>
              <ul className="space-y-2 text-slate-700">
                <li className="flex items-start gap-2">
                  <span className="text-indigo-600 mt-1">►</span>
                  <span>Market research analysis</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-indigo-600 mt-1">►</span>
                  <span>Competitor monitoring</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-indigo-600 mt-1">►</span>
                  <span>Customer segmentation insights</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Thesis Contribution */}
        <div className="mt-8 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl shadow-lg p-8 text-white">
          <h2 className="text-2xl font-bold mb-4">Your Thesis Contribution</h2>
          <p className="text-lg leading-relaxed">
            By systematically exploring different combinations of word embedding techniques, dimensionality reduction methods, 
            and SOM hyperparameters, your research identifies the optimal configuration for textual clusterization. 
            This provides a comprehensive framework for practitioners to select the best approach based on their 
            specific dataset characteristics and clustering objectives.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ThesisDiagram;
