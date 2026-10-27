import { useEffect, useMemo } from "react";
import L from "leaflet";
import { MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import type { TravelPlanDraft } from "@/api/types";
import { DAY_COLORS, dayRoutes, type Point } from "@/lib/places";

// 高德瓦片：国内访问稳定。坐标系与地点库一致时偏差很小，规划演示足够。
const TILE_URL = "https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}";

function icon(label: string, color: string, hotel = false) {
  return L.divIcon({
    className: "",
    html: `<div class="wp-marker" style="background:${color}">${hotel ? "🏨" : label}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
}

function FitBounds({ points }: { points: Point[] }) {
  const map = useMap();
  useEffect(() => {
    if (!points.length) return;
    const bounds = L.latLngBounds(points.map((p) => [p.lat, p.lng] as [number, number]));
    map.fitBounds(bounds, { padding: [36, 36], maxZoom: 14 });
  }, [map, points]);
  return null;
}

export function PlanMap({ plan, focusDay }: { plan: TravelPlanDraft; focusDay: number | null }) {
  const routes = useMemo(() => dayRoutes(plan), [plan]);
  const visible = useMemo(() => (focusDay ? routes.filter((r) => r.day === focusDay) : routes), [routes, focusDay]);
  const allPoints = useMemo(() => visible.flatMap((r) => r.points), [visible]);
  const hotel = routes[0]?.hotel;

  if (!routes.some((r) => r.points.length)) {
    return (
      <div className="flex h-full items-center justify-center rounded-2xl bg-ink-50 p-8 text-center text-sm text-ink-400 dark:bg-ink-800">
        这个城市的地点还没有坐标数据，暂时无法在地图上展示。
      </div>
    );
  }

  return (
    <MapContainer center={[allPoints[0]?.lat || 30.6, allPoints[0]?.lng || 104.06]} zoom={12} scrollWheelZoom>
      <TileLayer url={TILE_URL} subdomains={["1", "2", "3", "4"]} attribution="&copy; 高德地图" />
      <FitBounds points={allPoints} />
      {visible.map((route) => {
        const color = DAY_COLORS[(route.day - 1) % DAY_COLORS.length];
        const stops = route.hotel ? route.points.slice(1) : route.points;
        return (
          <div key={route.day}>
            <Polyline
              positions={route.points.map((p) => [p.lat, p.lng] as [number, number])}
              pathOptions={{ color, weight: 4, opacity: 0.85, dashArray: "8 8" }}
            />
            {stops.map((point, index) => (
              <Marker key={`${route.day}-${point.name}`} position={[point.lat, point.lng]} icon={icon(String(index + 1), color)}>
                <Tooltip direction="top" offset={[0, -12]}>
                  第{route.day}天 · {point.name}
                </Tooltip>
              </Marker>
            ))}
          </div>
        );
      })}
      {hotel && (
        <Marker position={[hotel.lat, hotel.lng]} icon={icon("", "#141b2e", true)}>
          <Tooltip direction="top" offset={[0, -12]}>
            住宿 · {hotel.name}
          </Tooltip>
        </Marker>
      )}
    </MapContainer>
  );
}
