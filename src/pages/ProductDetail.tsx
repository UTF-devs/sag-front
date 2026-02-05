import { useEffect, useState } from "react";
import { ContactInfoSection } from "../screens/HomePage/sections/ContactInfoSection";
import { ChevronLeft } from "lucide-react";
import { Link, useLocation, useParams, useNavigate } from "react-router-dom";
import { useLanguage } from "../contexts/LanguageContext";
import Characteristics from "../components/Characteristics";
import PurchaseModal from "../screens/PurchaseModal";
import { client } from "../services";
import ShareButtons from "../components/ShareButtons";

interface ProductImage {
  id: number;
  image: string;
}

interface CharacterDetail {
  id: number;
  character: string;
  detail: string;
}

interface ShapeDetail {
  id: number;
  shape: string;
  size: string;
  price: number;
}

interface ProductData {
  id: number;
  name: string;
  color: string;
  image: string;
  images: ProductImage[];
  character: string;
  character_details: CharacterDetail[];
  shapes: { [key: string]: ShapeDetail[] };
  catalog: { id: number; name: string }; // Added catalog
  model: { id: number; name: string }; // Added model
}

const ProductDetail = () => {
  const [product, setProduct] = useState<ProductData | null>(null);
  const [allImages, setAllImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<string>("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedShape, setSelectedShape] = useState<string>("");
  const location = useLocation();
  const { id } = useParams<{ id: string }>();
  const { t, language } = useLanguage();
  const navigate = useNavigate();
  const fromSales = location.state?.from === "sales";

  const mapLang = (lang: string) =>
    lang === "rus" ? "ru" : lang === "uzb" ? "uz" : "en";

  useEffect(() => {
    const lang = mapLang(language);
    const fetchData = async () => {
      try {
        const res = await client.get(
          `/${lang}/api/v1/catalog/get_carpet_model_by_id/${id}/`,
        );
        const data = res.data;
        setProduct(data);

        const imgArray = Array.isArray(data.images)
          ? data.images.map((img: ProductImage) => img.image)
          : [];
        const combined = [...imgArray, data.image];
        setAllImages(combined);
        setSelectedImage(combined[0] || "");

        if (data.shapes && Object.keys(data.shapes).length > 0) {
          setSelectedShape(Object.keys(data.shapes)[0]);
        }
      } catch (err) {
        console.error("Mahsulotni yuklashda xatolik:", err);
      }
    };

    if (id) fetchData();
  }, [id, language]);

  useEffect(() => {
    if (!allImages.length) return;

    const interval = setInterval(() => {
      setSelectedImage((prev) => {
        const index = allImages.indexOf(prev);
        const nextIndex = (index + 1) % allImages.length;
        return allImages[nextIndex];
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [allImages]);

  const translateShape = (shape: string) => {
    switch (shape) {
      case "Прямоугольник":
        return t("product.shape.rectangle") || "To‘rtburchak";
      case "Овал":
        return t("product.shape.oval") || "Oval";
      default:
        return shape;
    }
  };

  return (
    <div className="bg-[#FFFCE0] md:pt-32 pt-28">
      {/* Breadcrumbs */}
      <div className="flex flex-wrap items-center container mx-auto text-base text-gray-600 mb-4 px-4">
        <ChevronLeft size={20} className="text-gray-600" />
        <Link to="/">{t("nav.home")}</Link>
        {fromSales ? (
          <>
            <Link to="/sales" className="pl-3 flex items-center">
              <ChevronLeft size={20} className="text-gray-600" />
              {t("nav.sales")}
            </Link>
            <div
              onClick={() => window.history.back()}
              className="pl-3 cursor-pointer flex items-center"
            >
              <ChevronLeft size={20} className="text-gray-600" />
              {product?.name || t("common.loading")}
            </div>
            <div className="pl-3 flex items-center font-semibold">
              <ChevronLeft size={20} className="text-gray-600" />
              {t("product.breadcrumb.product")}
            </div>
          </>
        ) : (
          <>
            <div
              className="pl-3 flex items-center cursor-pointer"
              onClick={() => window.history.back()}
            >
              <ChevronLeft size={20} className="text-gray-600" />
              {product?.catalog?.name || t("product.breadcrumb.carpets")}
            </div>
            <div
              className="pl-3 flex items-center cursor-pointer"
              onClick={() => window.history.back()}
            >
              <ChevronLeft size={20} className="text-gray-600" />
              {product?.model?.name || t("product.breadcrumb.collection")}
            </div>
            <div className="pl-3 flex items-center font-semibold">
              <ChevronLeft size={20} className="text-gray-600" />
              {product?.name || t("product.breadcrumb.product")}
            </div>
          </>
        )}
      </div>

      {/* Mahsulot kontenti */}
      <div className="container min-h-[500px] mx-auto px-4 py-8">
        {!product ? (
          <p className="text-center text-gray-500">
            {t("common.loading") || "Yuklanmoqda..."}
          </p>
        ) : (
          <div className="md:flex gap-4 w-full">
            {/* Thumbnails */}
            <div className="flex md:flex-col gap-2 overflow-x-auto lg:overflow-visible">
              {allImages.map((img, idx) => (
                <img
                  key={idx}
                  src={img}
                  onClick={() => setSelectedImage(img)}
                  className={`w-20 h-20 object-cover cursor-pointer border ${
                    selectedImage === img
                      ? "border-blue-600"
                      : "border-gray-300"
                  }`}
                  alt={`thumb-${idx}`}
                />
              ))}
            </div>

            {/* Main Image */}
            <div className="flex-1 flex flex-col items-center md:pt-0 pt-9">
              <img
                src={selectedImage}
                alt="Product"
                className="w-full h-[500px] object-contain rounded"
              />
              <div className="flex justify-center gap-2 mt-3">
                {allImages.map((img, idx) => (
                  <div
                    key={idx}
                    className={`w-2 h-2 rounded-full ${
                      selectedImage === img ? "bg-blue-600" : "bg-gray-300"
                    }`}
                  ></div>
                ))}
              </div>
              <button
                className="mt-4 px-6 py-2 bg-[#CDAA7D] hover:bg-[#b8986d] text-white font-semibold rounded transition-colors"
                onClick={() => navigate(`/carpet-view/${id}`, {
                  state: {
                    carpet: {
                      id: product?.id || 0,
                      name: product?.name || "",
                      image: selectedImage,
                      price: product?.shapes && selectedShape && product.shapes[selectedShape] 
                        ? `${product.shapes[selectedShape][0]?.price.toLocaleString()} ${t("currency")}` 
                        : "",
                      description: product?.character || "",
                    }
                  }
                })}
              >
                {t("product.try_on")}
              </button>
              <div className="mt-5">
                <ShareButtons />
              </div>
            </div>

            {/* Info */}
            <div className="md:w-[40%] md:pt-0 pt-9">
              <h1 className="text-xl font-semibold text-gray-800 mb-2">
                {product.name}
              </h1>
              <p className="text-sm text-gray-500 mb-4">
                {t("filter.colors")}: {product.color}
              </p>

              {/* Shakl tanlash */}
              {product.shapes && Object.keys(product.shapes).length > 0 && (
                <div className="flex gap-4 mb-4">
                  {Object.keys(product.shapes).map((shape) => (
                    <button
                      key={shape}
                      className={`flex-1 py-2 rounded text-white font-semibold transition-colors ${
                        selectedShape === shape ? "bg-[#CDAA7D]" : "bg-gray-400"
                      }`}
                      onClick={() => setSelectedShape(shape)}
                    >
                      {translateShape(shape)}
                    </button>
                  ))}
                </div>
              )}

              {/* Narxlar jadvali */}
              {product.shapes &&
                selectedShape &&
                product.shapes[selectedShape] && (
                  <table className="w-full text-sm text-left mb-4">
                    <thead>
                      <tr className="border-b border-gray-300">
                        <th className="py-2">{t("product.sizes")}</th>
                        <th className="py-2">{t("product.price")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {product.shapes[selectedShape].map(
                        (item: ShapeDetail) => (
                          <tr key={item.id}>
                            <td className="py-2">{item.size}</td>
                            <td className="py-2 font-bold text-gray-800">
                              {item.price.toLocaleString()} {t("currency")}
                            </td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                )}

              {/* Xususiyatlar */}
              <div className="mb-4">
                <h2 className="text-lg font-semibold text-gray-800 mb-2">
                  {product.character}
                </h2>
                <Characteristics
                  details={product.character_details || []}
                  title={product.character}
                />
              </div>
            </div>
          </div>
        )}

        {/* Sotib olish tugmasi */}
        <button
          onClick={() => setIsModalOpen(true)}
          className="border mt-5 px-5 py-2 bg-blue-900 hover:bg-blue-800 rounded-md text-white"
        >
          {t("buy")}
        </button>

        <PurchaseModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
        />
      </div>

      <ContactInfoSection />
    </div>
  );
};

export default ProductDetail;
