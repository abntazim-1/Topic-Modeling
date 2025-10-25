# Topic Modeling Web Interface

A modern, responsive web interface for your Topic Modeling API with real-time analysis capabilities.

## 🌟 Features

### **Modern Dashboard**
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Status**: Live API connection and model status monitoring
- **Interactive UI**: Smooth animations and hover effects
- **Dark Mode Support**: Automatic dark mode detection

### **Model Management**
- **Load Models**: Load LDA and NMF models with one click
- **Status Indicators**: Visual status for each model (loaded/not loaded)
- **Topic Visualization**: View model topics with word weights
- **Model Information**: Detailed model metadata and statistics

### **Topic Prediction**
- **Single Text Analysis**: Analyze individual text documents
- **Batch Processing**: Process multiple texts simultaneously
- **Real-time Results**: Instant topic predictions with confidence scores
- **Interactive Charts**: Visual topic distribution and word clouds

### **User Experience**
- **Auto-save**: Prediction history saved locally
- **Keyboard Shortcuts**: Ctrl+Enter to submit predictions
- **Error Handling**: Comprehensive error messages and recovery
- **Loading States**: Visual feedback during processing

## 🚀 Getting Started

### **1. Start the API Server**
```bash
# Start the Flask API with web interface
python src/api/app.py

# Or with custom settings
python src/api/app.py --host 0.0.0.0 --port 5000 --debug
```

### **2. Access the Web Interface**
Open your browser and navigate to:
```
http://localhost:5000
```

### **3. Load Models**
1. Click "Load Model" for LDA and NMF models
2. Wait for the status to show "Loaded"
3. Models are now ready for prediction

### **4. Analyze Text**
1. Enter your text in the prediction form
2. Select model type (LDA or NMF)
3. Choose number of topics (1-20)
4. Click "Analyze Topics"

## 📱 Interface Overview

### **Navigation Bar**
- **Brand**: Topic Modeling Dashboard
- **Status**: API connection status (Connected/Disconnected)

### **Hero Section**
- **Title**: Topic Modeling Dashboard
- **Description**: Overview of capabilities
- **Action Buttons**: Quick access to analysis and model management

### **Features Section**
- **AI-Powered Analysis**: LDA and NMF algorithms
- **Real-time Visualization**: Interactive charts and graphs
- **Fast Processing**: Optimized for large datasets

### **Model Management**
- **LDA Model Card**: Load and manage LDA model
- **NMF Model Card**: Load and manage NMF model
- **Status Indicators**: Visual model status
- **Action Buttons**: Load models and view topics

### **Prediction Interface**
- **Text Input**: Large text area for document input
- **Model Selection**: Choose between LDA and NMF
- **Topic Count**: Set number of topics (1-20)
- **Submit Button**: Start analysis

### **Results Display**
- **Input Summary**: Original text preview
- **Model Information**: Model type and confidence
- **Topic Distribution**: Visual topic breakdown
- **Word Weights**: Top words per topic with scores

## 🎨 Customization

### **Styling**
The interface uses custom CSS with:
- **Bootstrap 5**: Modern responsive framework
- **Font Awesome**: Professional icons
- **Custom Animations**: Smooth transitions and effects
- **Gradient Backgrounds**: Modern visual design

### **JavaScript Features**
- **Dashboard Class**: Main application logic
- **Event Handling**: Form submissions and user interactions
- **API Communication**: RESTful API integration
- **Local Storage**: Prediction history persistence

### **Responsive Design**
- **Mobile First**: Optimized for mobile devices
- **Tablet Support**: Medium screen adaptations
- **Desktop Enhanced**: Full feature set on large screens

## 🔧 Configuration

### **API Configuration**
The web interface automatically connects to the Flask API. Configure in `config/config.yaml`:

```yaml
api:
  host: '0.0.0.0'
  port: 5000
  debug: false
  models_path: 'artifacts/topic_models/models'
```

### **Custom Styling**
Modify `src/api/static/css/style.css` for custom styling:
- Color schemes
- Animations
- Layout adjustments
- Responsive breakpoints

### **JavaScript Enhancements**
Extend `src/api/static/js/dashboard.js` for additional functionality:
- Custom API endpoints
- Advanced visualizations
- User preferences
- Data export features

## 📊 API Endpoints Used

### **Health Check**
```
GET /health
```
Returns API status and model availability.

### **Model Management**
```
GET /models
POST /models/{model_type}/load
GET /models/{model_type}/topics
```

### **Predictions**
```
POST /predict
POST /predict/batch
```

### **Visualization**
```
GET /models/{model_type}/visualize
```

## 🛠️ Development

### **File Structure**
```
src/api/
├── app.py                 # Flask application
├── routes.py             # Route utilities
├── templates/
│   └── index.html        # Main web interface
└── static/
    ├── css/
    │   └── style.css     # Custom styles
    └── js/
        └── dashboard.js   # Dashboard logic
```

### **Adding New Features**
1. **Backend**: Add new routes in `app.py`
2. **Frontend**: Update HTML in `templates/index.html`
3. **Styling**: Modify CSS in `static/css/style.css`
4. **Logic**: Extend JavaScript in `static/js/dashboard.js`

### **Testing**
```bash
# Start the API server
python src/api/app.py

# Test in browser
open http://localhost:5000

# Check API endpoints
curl http://localhost:5000/health
```

## 🚨 Troubleshooting

### **Common Issues**

1. **Models Not Loading**
   - Check if model files exist in `artifacts/topic_models/models/`
   - Verify model files are properly trained
   - Check API logs for error messages

2. **Web Interface Not Loading**
   - Ensure Flask app is running
   - Check static file paths
   - Verify template directory structure

3. **API Connection Issues**
   - Check if API server is running
   - Verify port configuration
   - Check firewall settings

4. **Prediction Errors**
   - Ensure models are loaded
   - Check input text format
   - Verify API endpoint responses

### **Debug Mode**
```bash
# Start with debug mode
python src/api/app.py --debug

# Check browser console for JavaScript errors
# Check API logs for backend errors
```

## 📈 Performance

### **Optimization Tips**
- **Model Loading**: Load models once at startup
- **Caching**: Implement prediction result caching
- **Batch Processing**: Use batch endpoints for multiple texts
- **Compression**: Enable gzip compression for static files

### **Monitoring**
- **API Health**: Regular health check monitoring
- **Model Status**: Track model loading and availability
- **User Analytics**: Monitor usage patterns and performance

## 🔒 Security

### **Best Practices**
- **Input Validation**: Sanitize user input
- **Rate Limiting**: Implement API rate limiting
- **CORS Configuration**: Proper cross-origin settings
- **Error Handling**: Secure error messages

### **Production Deployment**
- **HTTPS**: Use SSL certificates
- **Authentication**: Implement user authentication
- **Logging**: Comprehensive audit logging
- **Monitoring**: Real-time performance monitoring

## 📚 Documentation

### **API Documentation**
- **Swagger/OpenAPI**: Auto-generated API docs
- **Endpoint Testing**: Interactive API testing
- **Response Examples**: Sample API responses

### **User Guide**
- **Getting Started**: Step-by-step setup
- **Feature Overview**: Complete feature list
- **Troubleshooting**: Common issues and solutions

## 🎯 Future Enhancements

### **Planned Features**
- **User Authentication**: Login and user management
- **Project Management**: Save and load analysis projects
- **Advanced Visualizations**: Interactive topic networks
- **Export Features**: PDF and CSV report generation
- **Real-time Collaboration**: Multi-user analysis sessions

### **Integration Options**
- **Database Storage**: Persistent data storage
- **Cloud Deployment**: AWS/Azure/GCP deployment
- **Microservices**: Containerized service architecture
- **API Gateway**: Centralized API management

The web interface provides a complete, user-friendly solution for your topic modeling project with modern design and comprehensive functionality! 🚀
