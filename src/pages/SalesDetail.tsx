import { useEffect, useState } from "react";
import { useParams, Link, useLocation } from "react-router-dom";
import { ChevronLeft } from "lucide-react";
import { ContactInfoSection } from "../screens/HomePage/sections/ContactInfoSection";
import ProductCard from "../screens/SalesPage/ProductCardSales";
import { useLanguage } from "../contexts/LanguageContext";
import { client } from "../services";

interface Rug {
  id: number;
  name: string;
  price: number;
  originalPrice?: number | null;
  image: string;
  collection_type: number;
}

const SalesDetail = () => {
  const { t, language } = useLanguage();
  const { id } = useParams<{ id: string }>();
  const [roomName, setRoomName] = useState<string>("");
  const [rugData, setRugData] = useState<Rug[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const location = useLocation();
  const [error, setError] = useState<string | null>(null);
  const itemsPerPage = 21;

  const mapLang = (lang: string) =>
    lang === "rus" ? "ru" : lang === "uzb" ? "uz" : "en";

  useEffect(() => {
    const lang = mapLang(language);
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await client.get(
          `/${lang}/api/v1/discount/get_main_discounted_carpet_model/${id}/`,
        );

        setRoomName(
          location.state?.name || t("noName") || "Xona nomi mavjud emas",
        );
        // Agar ma'lumotlar "results" ichida bo'lsa yoki array bo'lmasa
        const items = Array.isArray(res.data)
          ? res.data
          : res.data.results || [];
        setRugData(items);
      } catch (err) {
        console.error("API xatosi:", err);
        setError(
          t("catalog.noProducts") ||
            "Ma'lumotlarni yuklashda xatolik yuz berdi",
        );
        setRugData([]);
        setRoomName(
          location.state?.name || t("noName") || "Xona nomi mavjud emas",
        );
      } finally {
        setLoading(false);
      }
    };

    if (id) fetchData();
  }, [id, language, t]);

  const totalPages = Math.ceil(rugData.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentRugs = rugData.slice(startIndex, startIndex + itemsPerPage);

  return (
    <div className="bg-[#FFFCE0] md:pt-28 pt-24">
      <Navbar />
      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <div className="flex items-center text-base text-gray-600 mb-4">
            <ChevronLeft size={20} className="text-gray-600" />
            <Link to="/">{t("nav.home") || "Bosh sahifa"}</Link>
            <Link to="/sales" className="pl-3 flex items-center">
              <ChevronLeft size={20} className="text-gray-600" />
              {t("nav.sales") || "Aksiyalar"}
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
              <p className="text-gray-500">
                {t("common.loading") || "Yuklanmoqda..."}
              </p>
            </div>
          ) : error ? (
            <div className="text-center py-12 text-red-500">{error}</div>
          ) : rugData.length > 0 ? (
            <>
              <div className="grid grid-cols-2  sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {currentRugs.map((rug) => (
                  <ProductCard
                    key={rug.id}
                    id={rug.id}
                    name={rug.name}
                    image={rug.image}
                    price={rug.price}
                    originalPrice={rug.originalPrice}
                    collectionType={rug.collection_type}
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
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                    (page) => (
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
                    ),
                  )}
                  <button
                    onClick={() =>
                      setCurrentPage(Math.min(totalPages, currentPage + 1))
                    }
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
              {/* <p className="text-gray-500">{t("common.noProducts") || "Hech qanday mahsulot topilmadi"}</p> */}
            </div>
          )}
        </div>
      </div>
      <ContactInfoSection />
    </div>
  );
};

export default SalesDetail;
