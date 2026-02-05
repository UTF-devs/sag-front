import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { ChevronLeft } from "lucide-react";
import { useLocation } from "react-router-dom";
import { ContactInfoSection } from "../screens/HomePage/sections/ContactInfoSection";
import { useLanguage } from "../contexts/LanguageContext";
import AppartmentProducts from "../screens/AppartmentPage/AppartmentProducts";
import { client } from "../services";

interface Product {
  id: number;
  name: string;
  price: number;
  originalPrice?: number | null;
  isNew?: boolean;
  isOnSale?: boolean;
  image: string;
}

const AppartmentDetail = () => {
  const { id } = useParams<{ id: string }>();
  const { t, language } = useLanguage();
  const [products, setProducts] = useState<Product[]>([]);
  const [roomName, setRoomName] = useState<string>("");
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const location = useLocation();
  const itemsPerPage = 21;

  const mapLang = (lang: string) =>
    lang === "rus" ? "ru" : lang === "uzb" ? "uz" : "en";

useEffect(() => {
  const lang = mapLang(language);

  const fetchRoomDetails = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await client.get(`/${lang}/api/v1/rooms/get_rooms_images_by_room_id/${id}/`);
      setProducts(response.data);

      // roomName ni backenddan kelgan ma'lumotdan olish
      const room = response.data?.[0]?.room?.name;
      setRoomName(room || t("noName") || "Xona nomi mavjud emas");

    } catch (err) {
      console.error("Ma'lumotlarni yuklashda xatolik:", err);
      setError(t("errors.fetchData") || "Ma'lumotlarni yuklashda xatolik yuz berdi");
      setRoomName(t("noName") || "Xona nomi mavjud emas");
    } finally {
      setLoading(false);
    }
  };

  if (id) fetchRoomDetails();
}, [id, language, t]);


  const totalPages = Math.ceil(products.length / itemsPerPage);
  const currentRugs = products.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  return (
    <div className="bg-[#FFFCE0] md:pt-28 pt-24">
      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <div className="flex items-center text-base text-gray-600 mb-4">
            <ChevronLeft size={20} className="text-gray-600" />
            <Link to="/">{t("nav.home") || "Bosh sahifa"}</Link>
            <Link to="/rooms" className="pl-3 flex items-center">
              <ChevronLeft size={20} className="text-gray-600" />
              {t("nav.rooms") || "Xonalar"}
            </Link>
           <div className="pl-3 flex items-center font-semibold">
  <ChevronLeft size={20} className="text-gray-600" />
  {roomName}
</div>

          </div>
        </div>

        <div className="w-full min-h-[500px]">
          {loading ? (
            <div className="text-center py-12">
              <p className="text-gray-500">{t("common.loading") || "Yuklanmoqda..."}</p>
            </div>
          ) : error ? (
            <div className="text-center py-12 text-red-500">{error}</div>
          ) : products.length > 0 ? (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {currentRugs.map((rug) => (
                  <AppartmentProducts
                    key={rug.id}
                    id={rug.id}
                    name={rug.name}
                    image={rug.image}
                    price={rug.price}
                    originalPrice={rug.originalPrice}
                    isNew={rug.isNew}
                    isOnSale={rug.isOnSale}
                  />
                ))}
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex justify-center items-center gap-2 mt-8">
                  <button
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1}
                    className="px-3 py-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-50"
                  >
                    ←
                  </button>

                  {[...Array(totalPages)].map((_, i) => {
                    const page = i + 1;
                    return (
                      <button
                        key={page}
                        onClick={() => setCurrentPage(page)}
                        className={`px-3 py-2 rounded ${
                          currentPage === page
                            ? "bg-[#D7CCC8] text-white"
                            : "text-gray-600 hover:bg-gray-100"
                        }`}
                      >
                        {page}
                      </button>
                    );
                  })}

                  <button
                    onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                    disabled={currentPage === totalPages}
                    className="px-3 py-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-50"
                  >
                    →
                  </button>
                </div>
              )}
            </>
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

export default AppartmentDetail;