import { useNavigate, useLocation } from "react-router-dom";
import { ChevronLeft } from "lucide-react";
import { Link } from "react-router-dom";
import CarpetView from "../components/CarpetView";
import { ContactInfoSection } from "../screens/HomePage/sections/ContactInfoSection";
import { useLanguage } from "../contexts/LanguageContext";
import type { Carpet } from "../types/carpet";

const CarpetViewPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useLanguage();

  const carpetData = location.state?.carpet as Carpet | undefined;

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
        {carpetData ? (
          <CarpetView carpet={carpetData} onChangeCarpet={() => navigate(-2)} />
        ) : (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <p className="text-gray-600 mb-4">
                {t("common.error") || "Gilam ma'lumotlari topilmadi"}
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
