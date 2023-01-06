mapboxgl.accessToken = 'pk.eyJ1IjoidmljdG9yYm5udCIsImEiOiJja29mams1Zm8wNmV5Mm9wbGtnZ21ybjVmIn0.xMyi36ZqeCDl4IKTQpIlCA';
const taxiFareApiUrl = 'https://moqku.co/taxifareapi/';

const map = new mapboxgl.Map({
  container: 'map',
  // Choose from Mapbox's core styles, or make your own style with Mapbox Studio
  style: 'mapbox://styles/mapbox/streets-v12',
  center: [-74.00597, 40.71427],
  zoom: 12
});

new_direction = new MapboxDirections({
  accessToken: mapboxgl.accessToken,
  profile: 'mapbox/driving',
  controls: {profileSwitcher: false,
             instructions: false}
})

map.addControl(new_direction, 'top-left');

const initFlatpickr = () => {
  flatpickr("#pickup_datetime", {
    enableTime: true,
    dateFormat: "Y-m-d H:i:S",
    defaultDate: Date.now()
  });
};
