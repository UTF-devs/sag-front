import { useEffect, useState } from "react";
import { useNavigate, useLocation, useParams } from "react-router-dom";
import { ChevronLeft } from "lucide-react";
import { Link } from "react-router-dom";
import CarpetView from "../components/CarpetView";
import { ContactInfoSection } from "../screens/HomePage/sections/ContactInfoSection";
import { useLanguage } from "../contexts/LanguageContext";
import { client } from "../services";
import type { Carpet } from "../types/carpet";

const CarpetViewPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { id } = useParams<{ id: string }>();
  const { t, language } = useLanguage();
  const [carpetData, setCarpetData] = useState<Carpet | null>(
    (location.state?.carpet as Carpet) ?? null
  );
  const [isLoading, setIsLoading] = useState(false);
  const [fetchError, setFetchError] = useState(false);

  const mapLang = (lang: string) =>
    lang === "rus" ? "ru" : lang === "uzb" ? "uz" : "en";

  // Fetch carpet by ID when navigating directly or when URL id differs from current carpet
  useEffect(() => {
    if (!id) return;
    const idNum = Number(id);
    if (carpetData && carpetData.id === idNum) return;
    let cancelled = false;
    setIsLoading(true);
    setFetchError(false);
    if (carpetData && carpetData.id !== idNum) setCarpetData(null);
    const lang = mapLang(language);
    client
      .get(`/${lang}/api/v1/catalog/get_carpet_model_by_id/${id}/`)
      .then((res) => {
        if (cancelled) return;
        const data = res.data;
        const imgArray = Array.isArray(data.images)
          ? data.images.map((img: { image: string }) => img.image)
          : [];
        const mainImage = imgArray[0] || data.image;
        const firstShapeKey = data.shapes && Object.keys(data.shapes)[0];
        const priceStr =
          firstShapeKey && data.shapes[firstShapeKey]?.[0]?.price != null
            ? `${data.shapes[firstShapeKey][0].price.toLocaleString()} ${t("currency")}`
            : "";
        setCarpetData({
          id: data.id,
          name: data.name || "",
          image: mainImage,
          price: priceStr,
          description: data.character || "",
        });
      })
      .catch(() => {
        if (!cancelled) setFetchError(true);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id, language, t, carpetData]);

  return (
    <div className="bg-[#FFFCE0] md:pt-32 pt-28">
      {/* Breadcrumbs */}
      <div className="flex flex-wrap items-center container mx-auto text-base text-gray-600 mb-4 px-4">
        <ChevronLeft size={20} className="text-gray-600" />
        <Link to="/">{t("nav.home")}</Link>
        <div
          onClick={() => navigate(-1)}
          className="pl-3 cursor-pointer flex items-center"
        >
          <ChevronLeft size={20} className="text-gray-600" />
          {t("product.breadcrumb.product")}
        </div>
        <div className="pl-3 flex items-center font-semibold">
          <ChevronLeft size={20} className="text-gray-600" />
          {t("product.try_on")}
        </div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 py-8 min-h-[600px]">
        {isLoading ? (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <p className="text-gray-600 mb-4">
                {t("common.loading") || "Yuklanmoqda..."}
              </p>
            </div>
          </div>
        ) : carpetData ? (
          <CarpetView carpet={carpetData} onChangeCarpet={() => navigate(-2)} />
        ) : (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <p className="text-gray-600 mb-4">
                {fetchError || !id
                  ? t("common.error")
                  : "Gilam ma'lumotlari topilmadi"}
              </p>
              <button
                onClick={() => navigate(-1)}
                className="px-6 py-2 bg-[#CDAA7D] hover:bg-[#b8986d] text-white font-semibold rounded transition-colors"
              >
                {t("video.back") || "Orqaga"}
              </button>
            </div>
          </div>
        )}
      </div>

      <ContactInfoSection />
    </div>
  );
};

export default CarpetViewPage;
