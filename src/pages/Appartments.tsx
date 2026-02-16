import { useEffect, useState } from "react";
import { client } from "../services";
import AppartmentCategory from "../screens/AppartmentPage/AppartmentCategory";
import { ContactInfoSection } from "../screens/HomePage/sections/ContactInfoSection";
import { ChevronLeft } from "lucide-react";
import { Link } from "react-router-dom";
import { useLanguage } from "../contexts/LanguageContext";

interface Room {
  id: number;
  name: string;
  image: string;
}

const Appartments = () => {
  const { t, language } = useLanguage();
  const [rooms, setRooms] = useState<Room[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const mapLang = (lang: string) =>
    lang === "rus" ? "ru" : lang === "uzb" ? "uz" : "en";

  useEffect(() => {
    const lang = mapLang(language);
    const fetchRooms = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await client.get(`/${lang}/api/v1/rooms/get_rooms/`);
        setRooms(res.data);
      } catch (error) {
        console.error("Xonalarni yuklashda xatolik:", error);
        setError(t("errors.fetchRooms") || "Xonalarni yuklashda xatolik yuz berdi");
      } finally {
        setLoading(false);
      }
    };

    fetchRooms();
  }, [language, t]);

  return (
    <div className="bg-[#FFFCE0] md:pt-28 pt-24">
      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <div className="flex items-center text-base text-gray-600 mb-4">
            <ChevronLeft size={20} className="text-gray-600" />
            <Link to="/">{t("nav.home") || "Bosh sahifa"}</Link>
            <div className="pl-3 flex items-center font-semibold">
              <ChevronLeft size={20} className="text-gray-600" />
              {t("nav.rooms") || "Xonalar"}
            </div>
          </div>
        </div>

        <div className="w-full min-h-[500px]">
          {loading ? (
            <div className="text-center py-12">
              <p className="text-gray-500">{t('common.loading')}</p>
            </div>
          ) : error ? (
            <div className="text-center py-12 text-red-500">{error}</div>
          ) : rooms.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {rooms.map((room) => (
                <AppartmentCategory
                  key={room.id}
                  id={room.id}
                  name={room.name}
                  image={room.image}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-gray-500"></p>
            </div>
          )}
        </div>
      </div>

      <ContactInfoSection />
    </div>
  );
};

export default Appartments;