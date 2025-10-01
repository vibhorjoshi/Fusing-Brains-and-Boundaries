'use client';

import React, { useEffect, useRef, useState } from 'react';

declare global {
  interface Window {
    google: any;
    initializeMap?: () => void;
  }
}

interface CityData {
  name: string;
  lat: number;
  lng: number;
  buildings: number;
  accuracy: number;
  processingTime: number;
}

const AlabamaMap: React.FC = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [selectedCity, setSelectedCity] = useState<string>('');
  const [visualization, setVisualization] = useState('markers');

  const alabamaCities: CityData[] = [
    { name: "Birmingham", lat: 33.5207, lng: -86.8025, buildings: 156421, accuracy: 0.912, processingTime: 26.4 },
    { name: "Montgomery", lat: 32.3792, lng: -86.3077, buildings: 98742, accuracy: 0.897, processingTime: 18.7 },
    { name: "Mobile", lat: 30.6954, lng: -88.0399, buildings: 87634, accuracy: 0.884, processingTime: 17.2 },
    { name: "Huntsville", lat: 34.7304, lng: -86.5861, buildings: 124563, accuracy: 0.923, processingTime: 22.9 },
    { name: "Tuscaloosa", lat: 33.2098, lng: -87.5692, buildings: 65432, accuracy: 0.901, processingTime: 14.8 },
    { name: "Auburn", lat: 32.6099, lng: -85.4808, buildings: 42387, accuracy: 0.894, processingTime: 10.5 },
    { name: "Dothan", lat: 31.2232, lng: -85.3905, buildings: 38921, accuracy: 0.878, processingTime: 9.7 },
    { name: "Hoover", lat: 33.4054, lng: -86.8114, buildings: 47532, accuracy: 0.907, processingTime: 12.3 },
    { name: "Decatur", lat: 34.6059, lng: -86.9834, buildings: 36281, accuracy: 0.889, processingTime: 8.9 },
    { name: "Madison", lat: 34.6992, lng: -86.7484, buildings: 41756, accuracy: 0.917, processingTime: 10.8 }
  ];

  useEffect(() => {
    const loadGoogleMaps = () => {
      if (window.google && window.google.maps) {
        initializeMap();
        return;
      }

      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=visualization&callback=initializeMap`;
      script.async = true;
      script.defer = true;
      
      window.initializeMap = initializeMap;
      document.head.appendChild(script);
    };

    loadGoogleMaps();
  }, []);

  const initializeMap = () => {
    if (!mapRef.current || !window.google) return;

    const alabamaCenter = { lat: 32.7794, lng: -86.8287 };
    
    const map = new window.google.maps.Map(mapRef.current, {
      center: alabamaCenter,
      zoom: 7,
      mapTypeId: 'hybrid',
      styles: [
        { elementType: 'geometry', stylers: [{ color: '#242f3e' }] },
        { elementType: 'labels.text.stroke', stylers: [{ color: '#242f3e' }] },
        { elementType: 'labels.text.fill', stylers: [{ color: '#746855' }] },
        { featureType: 'administrative.locality', elementType: 'labels.text.fill', stylers: [{ color: '#d59563' }] },
        { featureType: 'poi', elementType: 'labels.text.fill', stylers: [{ color: '#d59563' }] },
        { featureType: 'poi.park', elementType: 'geometry', stylers: [{ color: '#263c3f' }] },
        { featureType: 'poi.park', elementType: 'labels.text.fill', stylers: [{ color: '#6b9a76' }] },
        { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#38414e' }] },
        { featureType: 'road', elementType: 'geometry.stroke', stylers: [{ color: '#212a37' }] },
        { featureType: 'road', elementType: 'labels.text.fill', stylers: [{ color: '#9ca5b3' }] },
        { featureType: 'road.highway', elementType: 'geometry', stylers: [{ color: '#746855' }] },
        { featureType: 'road.highway', elementType: 'geometry.stroke', stylers: [{ color: '#1f2835' }] },
        { featureType: 'road.highway', elementType: 'labels.text.fill', stylers: [{ color: '#f3d19c' }] },
        { featureType: 'transit', elementType: 'geometry', stylers: [{ color: '#2f3948' }] },
        { featureType: 'transit.station', elementType: 'labels.text.fill', stylers: [{ color: '#d59563' }] },
        { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#17263c' }] },
        { featureType: 'water', elementType: 'labels.text.fill', stylers: [{ color: '#515c6d' }] },
        { featureType: 'water', elementType: 'labels.text.stroke', stylers: [{ color: '#17263c' }] }
      ]
    });

    const infoWindow = new window.google.maps.InfoWindow();
    const markers: any[] = [];
    
    // Create markers for each city
    alabamaCities.forEach(city => {
      const hue = (city.accuracy - 0.85) * 1200;
      const color = `hsl(${hue}, 100%, 50%)`;
      
      const marker = new window.google.maps.Marker({
        position: { lat: city.lat, lng: city.lng },
        map: map,
        title: city.name,
        icon: {
          path: window.google.maps.SymbolPath.CIRCLE,
          scale: 10,
          fillColor: color,
          fillOpacity: 0.7,
          strokeWeight: 1,
          strokeColor: '#ffffff'
        }
      });
      
      const contentString = `
        <div class="info-window text-white">
          <h3 class="text-lg font-bold">${city.name}, Alabama</h3>
          <p><strong>Buildings Detected:</strong> ${city.buildings.toLocaleString()}</p>
          <p><strong>Detection Accuracy:</strong> ${(city.accuracy * 100).toFixed(1)}%</p>
          <p><strong>Processing Time:</strong> ${city.processingTime.toFixed(1)} seconds</p>
        </div>
      `;
      
      marker.addListener('click', () => {
        infoWindow.setContent(contentString);
        infoWindow.open(map, marker);
        setSelectedCity(city.name);
      });
      
      markers.push(marker);
    });

    // Create heatmap
    const heatmapData = alabamaCities.map(city => ({
      location: new window.google.maps.LatLng(city.lat, city.lng),
      weight: city.buildings / 10000
    }));
    
    const heatmap = new window.google.maps.visualization.HeatmapLayer({
      data: heatmapData,
      map: null,
      radius: 30,
      opacity: 0.7
    });

    // Update visualization based on state
    const updateVisualization = (value: string) => {
      markers.forEach(marker => marker.setVisible(false));
      heatmap.setMap(null);
      
      if (value === 'markers') {
        markers.forEach(marker => marker.setVisible(true));
      } else if (value === 'heatmap') {
        heatmap.setMap(map);
      } else if (value === 'both') {
        markers.forEach(marker => marker.setVisible(true));
        heatmap.setMap(map);
      }
    };

    updateVisualization(visualization);
    setIsLoaded(true);
  };

  return (
    <div className="w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      <div className="p-4 bg-gray-800 border-b border-gray-700">
        <h2 className="text-xl font-bold text-white mb-4">Alabama Cities GeoAI Analysis</h2>
        
        <div className="flex gap-4 items-center">
          <label className="text-gray-300">Visualization:</label>
          <select 
            value={visualization}
            onChange={(e) => setVisualization(e.target.value)}
            className="bg-gray-700 text-white px-3 py-1 rounded border border-gray-600"
          >
            <option value="markers">Markers</option>
            <option value="heatmap">Heatmap</option>
            <option value="both">Both</option>
          </select>
          
          {selectedCity && (
            <div className="ml-auto text-blue-400">
              Selected: {selectedCity}
            </div>
          )}
        </div>
      </div>
      
      <div 
        ref={mapRef}
        className="w-full h-96"
        style={{ minHeight: '400px' }}
      />
      
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75">
          <div className="text-white">Loading Google Maps...</div>
        </div>
      )}
    </div>
  );
};

export default AlabamaMap;